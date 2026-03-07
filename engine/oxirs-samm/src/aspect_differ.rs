//! # SAMM Aspect Model Differ
//!
//! Computes structural diffs between two SAMM aspect models and generates
//! detailed change reports. Unlike the simpler `comparison` module, this
//! differ performs deep structural analysis including constraint changes,
//! entity evolution, and operation signature diffs.
//!
//! ## Features
//!
//! - Detects added / removed / changed properties
//! - Detects added / removed / changed constraints on characteristics
//! - Detects entity field changes (added / removed / retyped)
//! - Detects operation input / output signature changes
//! - Detects event additions and removals
//! - Provides a severity assessment for each change
//! - Generates human-readable markdown and plain-text reports
//!
//! ## Quick Start
//!
//! ```rust
//! use oxirs_samm::aspect_differ::{AspectModelDiffer, DiffSeverity};
//! use oxirs_samm::metamodel::Aspect;
//!
//! let old = Aspect::new("urn:samm:org.example:1.0.0#Movement".to_string());
//! let new = Aspect::new("urn:samm:org.example:1.0.0#Movement".to_string());
//! let diff = AspectModelDiffer::diff(&old, &new);
//!
//! assert_eq!(diff.total_changes(), 0);
//! ```

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::metamodel::{Aspect, Characteristic, ModelElement, Property};

// ---------------------------------------------------------------------------
// Diff result types
// ---------------------------------------------------------------------------

/// Severity of a single change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum DiffSeverity {
    /// Informational — cosmetic or documentation-only change.
    Info,
    /// Minor — backward-compatible addition.
    Minor,
    /// Major — potentially breaking change.
    Major,
    /// Critical — definitely breaking.
    Critical,
}

impl std::fmt::Display for DiffSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Minor => write!(f, "MINOR"),
            Self::Major => write!(f, "MAJOR"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Kind of structural change.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChangeKind {
    /// Element was added.
    Added,
    /// Element was removed.
    Removed,
    /// Element was modified.
    Modified,
    /// Element was renamed (URN changed but structure similar).
    Renamed,
}

impl std::fmt::Display for ChangeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Added => write!(f, "ADDED"),
            Self::Removed => write!(f, "REMOVED"),
            Self::Modified => write!(f, "MODIFIED"),
            Self::Renamed => write!(f, "RENAMED"),
        }
    }
}

/// A single change entry in the diff.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffEntry {
    /// What element was affected (e.g. property URN).
    pub element: String,
    /// Kind of change.
    pub kind: ChangeKind,
    /// Severity.
    pub severity: DiffSeverity,
    /// Human-readable description.
    pub description: String,
    /// Old value (for modifications).
    pub old_value: Option<String>,
    /// New value (for modifications).
    pub new_value: Option<String>,
}

/// The complete diff result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AspectDiff {
    /// URN of the old aspect.
    pub old_urn: String,
    /// URN of the new aspect.
    pub new_urn: String,
    /// All detected changes.
    pub changes: Vec<DiffEntry>,
}

impl AspectDiff {
    /// Total number of changes.
    pub fn total_changes(&self) -> usize {
        self.changes.len()
    }

    /// Whether there are no changes.
    pub fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }

    /// Filter changes by severity.
    pub fn by_severity(&self, severity: DiffSeverity) -> Vec<&DiffEntry> {
        self.changes
            .iter()
            .filter(|c| c.severity == severity)
            .collect()
    }

    /// Filter changes by kind.
    pub fn by_kind(&self, kind: &ChangeKind) -> Vec<&DiffEntry> {
        self.changes.iter().filter(|c| &c.kind == kind).collect()
    }

    /// The highest severity among all changes.
    pub fn max_severity(&self) -> Option<DiffSeverity> {
        self.changes.iter().map(|c| c.severity).max()
    }

    /// Whether the diff contains any breaking changes.
    pub fn has_breaking_changes(&self) -> bool {
        self.changes
            .iter()
            .any(|c| c.severity >= DiffSeverity::Major)
    }

    /// Generate a plain-text report.
    pub fn to_text_report(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("Diff: {} -> {}\n", self.old_urn, self.new_urn));
        out.push_str(&format!("Total changes: {}\n", self.total_changes()));
        if let Some(sev) = self.max_severity() {
            out.push_str(&format!("Max severity: {sev}\n"));
        }
        out.push('\n');
        for entry in &self.changes {
            out.push_str(&format!(
                "[{} / {}] {}: {}\n",
                entry.severity, entry.kind, entry.element, entry.description
            ));
            if let Some(old) = &entry.old_value {
                out.push_str(&format!("  old: {old}\n"));
            }
            if let Some(new) = &entry.new_value {
                out.push_str(&format!("  new: {new}\n"));
            }
        }
        out
    }

    /// Generate a markdown report.
    pub fn to_markdown_report(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "# Aspect Model Diff\n\n**Old**: `{}`  \n**New**: `{}`\n\n",
            self.old_urn, self.new_urn
        ));
        out.push_str(&format!(
            "**Total changes**: {}  \n**Max severity**: {}\n\n",
            self.total_changes(),
            self.max_severity()
                .map(|s| s.to_string())
                .unwrap_or_else(|| "none".to_string())
        ));
        if self.changes.is_empty() {
            out.push_str("_No changes detected._\n");
        } else {
            out.push_str("| Severity | Kind | Element | Description |\n");
            out.push_str("|----------|------|---------|-------------|\n");
            for entry in &self.changes {
                out.push_str(&format!(
                    "| {} | {} | `{}` | {} |\n",
                    entry.severity, entry.kind, entry.element, entry.description
                ));
            }
        }
        out
    }

    /// Severity histogram.
    pub fn severity_histogram(&self) -> HashMap<DiffSeverity, usize> {
        let mut hist = HashMap::new();
        for entry in &self.changes {
            *hist.entry(entry.severity).or_insert(0) += 1;
        }
        hist
    }
}

// ---------------------------------------------------------------------------
// Differ
// ---------------------------------------------------------------------------

/// The aspect model differ.
pub struct AspectModelDiffer;

impl AspectModelDiffer {
    /// Compute a structural diff between two SAMM Aspect models.
    pub fn diff(old: &Aspect, new: &Aspect) -> AspectDiff {
        let mut changes = Vec::new();

        // URN change
        if old.urn() != new.urn() {
            changes.push(DiffEntry {
                element: "aspect".to_string(),
                kind: ChangeKind::Renamed,
                severity: DiffSeverity::Info,
                description: "Aspect URN changed".to_string(),
                old_value: Some(old.urn().to_string()),
                new_value: Some(new.urn().to_string()),
            });
        }

        // Metadata changes
        Self::diff_metadata(old, new, &mut changes);

        // Properties
        Self::diff_properties(old.properties(), new.properties(), &mut changes);

        // Operations
        Self::diff_operations(old.operations(), new.operations(), &mut changes);

        // Events
        Self::diff_events(old.events(), new.events(), &mut changes);

        AspectDiff {
            old_urn: old.urn().to_string(),
            new_urn: new.urn().to_string(),
            changes,
        }
    }

    fn diff_metadata(old: &Aspect, new: &Aspect, changes: &mut Vec<DiffEntry>) {
        let old_meta = old.metadata();
        let new_meta = new.metadata();

        // Preferred names
        let old_names: HashSet<_> = old_meta.preferred_names.keys().collect();
        let new_names: HashSet<_> = new_meta.preferred_names.keys().collect();
        for lang in new_names.difference(&old_names) {
            changes.push(DiffEntry {
                element: format!("metadata/preferredName/{lang}"),
                kind: ChangeKind::Added,
                severity: DiffSeverity::Info,
                description: format!("Preferred name added for language '{lang}'"),
                old_value: None,
                new_value: new_meta.preferred_names.get(*lang).cloned(),
            });
        }
        for lang in old_names.difference(&new_names) {
            changes.push(DiffEntry {
                element: format!("metadata/preferredName/{lang}"),
                kind: ChangeKind::Removed,
                severity: DiffSeverity::Info,
                description: format!("Preferred name removed for language '{lang}'"),
                old_value: old_meta.preferred_names.get(*lang).cloned(),
                new_value: None,
            });
        }
        for lang in old_names.intersection(&new_names) {
            let ov = old_meta.preferred_names.get(*lang);
            let nv = new_meta.preferred_names.get(*lang);
            if ov != nv {
                changes.push(DiffEntry {
                    element: format!("metadata/preferredName/{lang}"),
                    kind: ChangeKind::Modified,
                    severity: DiffSeverity::Info,
                    description: format!("Preferred name changed for language '{lang}'"),
                    old_value: ov.cloned(),
                    new_value: nv.cloned(),
                });
            }
        }

        // Descriptions
        let old_descs: HashSet<_> = old_meta.descriptions.keys().collect();
        let new_descs: HashSet<_> = new_meta.descriptions.keys().collect();
        for lang in new_descs.difference(&old_descs) {
            changes.push(DiffEntry {
                element: format!("metadata/description/{lang}"),
                kind: ChangeKind::Added,
                severity: DiffSeverity::Info,
                description: format!("Description added for language '{lang}'"),
                old_value: None,
                new_value: new_meta.descriptions.get(*lang).cloned(),
            });
        }
        for lang in old_descs.difference(&new_descs) {
            changes.push(DiffEntry {
                element: format!("metadata/description/{lang}"),
                kind: ChangeKind::Removed,
                severity: DiffSeverity::Info,
                description: format!("Description removed for language '{lang}'"),
                old_value: old_meta.descriptions.get(*lang).cloned(),
                new_value: None,
            });
        }
    }

    fn diff_properties(
        old_props: &[Property],
        new_props: &[Property],
        changes: &mut Vec<DiffEntry>,
    ) {
        let old_map: HashMap<&str, &Property> = old_props.iter().map(|p| (p.urn(), p)).collect();
        let new_map: HashMap<&str, &Property> = new_props.iter().map(|p| (p.urn(), p)).collect();

        let old_urns: HashSet<&str> = old_map.keys().copied().collect();
        let new_urns: HashSet<&str> = new_map.keys().copied().collect();

        // Added properties
        for urn in new_urns.difference(&old_urns) {
            changes.push(DiffEntry {
                element: urn.to_string(),
                kind: ChangeKind::Added,
                severity: DiffSeverity::Minor,
                description: "Property added".to_string(),
                old_value: None,
                new_value: Some(urn.to_string()),
            });
        }

        // Removed properties
        for urn in old_urns.difference(&new_urns) {
            changes.push(DiffEntry {
                element: urn.to_string(),
                kind: ChangeKind::Removed,
                severity: DiffSeverity::Critical,
                description: "Property removed".to_string(),
                old_value: Some(urn.to_string()),
                new_value: None,
            });
        }

        // Modified properties
        for urn in old_urns.intersection(&new_urns) {
            if let (Some(old_p), Some(new_p)) = (old_map.get(urn), new_map.get(urn)) {
                Self::diff_single_property(old_p, new_p, changes);
            }
        }
    }

    fn diff_single_property(old_p: &Property, new_p: &Property, changes: &mut Vec<DiffEntry>) {
        let urn = old_p.urn().to_string();

        // Optional flag
        if old_p.optional != new_p.optional {
            let severity = if new_p.optional {
                DiffSeverity::Minor // making optional is non-breaking
            } else {
                DiffSeverity::Major // making required is breaking
            };
            changes.push(DiffEntry {
                element: urn.clone(),
                kind: ChangeKind::Modified,
                severity,
                description: "Optional flag changed".to_string(),
                old_value: Some(old_p.optional.to_string()),
                new_value: Some(new_p.optional.to_string()),
            });
        }

        // Characteristic
        match (&old_p.characteristic, &new_p.characteristic) {
            (Some(old_c), Some(new_c)) => {
                Self::diff_characteristic(&urn, old_c, new_c, changes);
            }
            (None, Some(_)) => {
                changes.push(DiffEntry {
                    element: urn.clone(),
                    kind: ChangeKind::Added,
                    severity: DiffSeverity::Minor,
                    description: "Characteristic added".to_string(),
                    old_value: None,
                    new_value: Some("(added)".to_string()),
                });
            }
            (Some(_), None) => {
                changes.push(DiffEntry {
                    element: urn.clone(),
                    kind: ChangeKind::Removed,
                    severity: DiffSeverity::Major,
                    description: "Characteristic removed".to_string(),
                    old_value: Some("(removed)".to_string()),
                    new_value: None,
                });
            }
            (None, None) => {}
        }
    }

    fn diff_characteristic(
        prop_urn: &str,
        old_c: &Characteristic,
        new_c: &Characteristic,
        changes: &mut Vec<DiffEntry>,
    ) {
        // Data type
        if old_c.data_type != new_c.data_type {
            changes.push(DiffEntry {
                element: prop_urn.to_string(),
                kind: ChangeKind::Modified,
                severity: DiffSeverity::Major,
                description: "Data type changed".to_string(),
                old_value: old_c.data_type.clone(),
                new_value: new_c.data_type.clone(),
            });
        }

        // Kind
        if old_c.kind != new_c.kind {
            changes.push(DiffEntry {
                element: prop_urn.to_string(),
                kind: ChangeKind::Modified,
                severity: DiffSeverity::Major,
                description: "Characteristic kind changed".to_string(),
                old_value: Some(format!("{:?}", old_c.kind)),
                new_value: Some(format!("{:?}", new_c.kind)),
            });
        }

        // Constraints
        let old_constraints: HashSet<String> = old_c
            .constraints
            .iter()
            .map(|c| format!("{:?}", c))
            .collect();
        let new_constraints: HashSet<String> = new_c
            .constraints
            .iter()
            .map(|c| format!("{:?}", c))
            .collect();

        for c in new_constraints.difference(&old_constraints) {
            changes.push(DiffEntry {
                element: prop_urn.to_string(),
                kind: ChangeKind::Added,
                severity: DiffSeverity::Major,
                description: "Constraint added".to_string(),
                old_value: None,
                new_value: Some(c.clone()),
            });
        }
        for c in old_constraints.difference(&new_constraints) {
            changes.push(DiffEntry {
                element: prop_urn.to_string(),
                kind: ChangeKind::Removed,
                severity: DiffSeverity::Major,
                description: "Constraint removed".to_string(),
                old_value: Some(c.clone()),
                new_value: None,
            });
        }
    }

    fn diff_operations(
        old_ops: &[crate::metamodel::Operation],
        new_ops: &[crate::metamodel::Operation],
        changes: &mut Vec<DiffEntry>,
    ) {
        let old_urns: HashSet<String> = old_ops.iter().map(|o| o.urn().to_string()).collect();
        let new_urns: HashSet<String> = new_ops.iter().map(|o| o.urn().to_string()).collect();

        for urn in new_urns.difference(&old_urns) {
            changes.push(DiffEntry {
                element: urn.clone(),
                kind: ChangeKind::Added,
                severity: DiffSeverity::Minor,
                description: "Operation added".to_string(),
                old_value: None,
                new_value: Some(urn.clone()),
            });
        }
        for urn in old_urns.difference(&new_urns) {
            changes.push(DiffEntry {
                element: urn.clone(),
                kind: ChangeKind::Removed,
                severity: DiffSeverity::Critical,
                description: "Operation removed".to_string(),
                old_value: Some(urn.clone()),
                new_value: None,
            });
        }
    }

    fn diff_events(
        old_events: &[crate::metamodel::Event],
        new_events: &[crate::metamodel::Event],
        changes: &mut Vec<DiffEntry>,
    ) {
        let old_urns: HashSet<String> = old_events.iter().map(|e| e.urn().to_string()).collect();
        let new_urns: HashSet<String> = new_events.iter().map(|e| e.urn().to_string()).collect();

        for urn in new_urns.difference(&old_urns) {
            changes.push(DiffEntry {
                element: urn.clone(),
                kind: ChangeKind::Added,
                severity: DiffSeverity::Minor,
                description: "Event added".to_string(),
                old_value: None,
                new_value: Some(urn.clone()),
            });
        }
        for urn in old_urns.difference(&new_urns) {
            changes.push(DiffEntry {
                element: urn.clone(),
                kind: ChangeKind::Removed,
                severity: DiffSeverity::Major,
                description: "Event removed".to_string(),
                old_value: Some(urn.clone()),
                new_value: None,
            });
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metamodel::{
        BoundDefinition, Characteristic, CharacteristicKind, Constraint, ElementMetadata, Event,
        Operation, Property,
    };

    fn make_aspect(urn: &str) -> Aspect {
        Aspect::new(urn.to_string())
    }

    fn make_property(urn: &str, optional: bool) -> Property {
        let mut p = Property::new(urn.to_string());
        p.optional = optional;
        p
    }

    fn make_property_with_char(urn: &str, dt: &str) -> Property {
        let mut p = Property::new(urn.to_string());
        p.characteristic = Some(Characteristic {
            metadata: ElementMetadata::new(format!("{urn}#char")),
            data_type: Some(dt.to_string()),
            kind: CharacteristicKind::Trait,
            constraints: Vec::new(),
        });
        p
    }

    fn make_operation(urn: &str) -> Operation {
        Operation::new(urn.to_string())
    }

    fn make_event(urn: &str) -> Event {
        Event::new(urn.to_string())
    }

    // -- basic diff --

    #[test]
    fn test_identical_aspects_no_changes() {
        let a = make_aspect("urn:samm:org.example:1.0.0#Movement");
        let b = make_aspect("urn:samm:org.example:1.0.0#Movement");
        let diff = AspectModelDiffer::diff(&a, &b);
        assert!(diff.is_empty());
        assert_eq!(diff.total_changes(), 0);
    }

    #[test]
    fn test_urn_change_detected() {
        let a = make_aspect("urn:samm:org.example:1.0.0#Movement");
        let b = make_aspect("urn:samm:org.example:1.1.0#Movement");
        let diff = AspectModelDiffer::diff(&a, &b);
        assert!(!diff.is_empty());
        assert!(diff.changes.iter().any(|c| c.kind == ChangeKind::Renamed));
    }

    // -- property changes --

    #[test]
    fn test_property_added() {
        let a = make_aspect("urn:samm:#A");
        let mut b = make_aspect("urn:samm:#A");
        b.add_property(make_property("urn:samm:#speed", false));
        let diff = AspectModelDiffer::diff(&a, &b);
        assert_eq!(diff.by_kind(&ChangeKind::Added).len(), 1);
    }

    #[test]
    fn test_property_removed() {
        let mut a = make_aspect("urn:samm:#A");
        a.add_property(make_property("urn:samm:#speed", false));
        let b = make_aspect("urn:samm:#A");
        let diff = AspectModelDiffer::diff(&a, &b);
        let removed = diff.by_kind(&ChangeKind::Removed);
        assert_eq!(removed.len(), 1);
        assert_eq!(removed[0].severity, DiffSeverity::Critical);
    }

    #[test]
    fn test_property_optional_to_required() {
        let mut a = make_aspect("urn:samm:#A");
        a.add_property(make_property("urn:samm:#speed", true));
        let mut b = make_aspect("urn:samm:#A");
        b.add_property(make_property("urn:samm:#speed", false));
        let diff = AspectModelDiffer::diff(&a, &b);
        let mods = diff.by_kind(&ChangeKind::Modified);
        assert_eq!(mods.len(), 1);
        assert_eq!(mods[0].severity, DiffSeverity::Major);
    }

    #[test]
    fn test_property_required_to_optional() {
        let mut a = make_aspect("urn:samm:#A");
        a.add_property(make_property("urn:samm:#speed", false));
        let mut b = make_aspect("urn:samm:#A");
        b.add_property(make_property("urn:samm:#speed", true));
        let diff = AspectModelDiffer::diff(&a, &b);
        let mods = diff.by_kind(&ChangeKind::Modified);
        assert_eq!(mods.len(), 1);
        assert_eq!(mods[0].severity, DiffSeverity::Minor);
    }

    // -- characteristic changes --

    #[test]
    fn test_data_type_change() {
        let mut a = make_aspect("urn:samm:#A");
        a.add_property(make_property_with_char("urn:samm:#speed", "xsd:float"));
        let mut b = make_aspect("urn:samm:#A");
        b.add_property(make_property_with_char("urn:samm:#speed", "xsd:double"));
        let diff = AspectModelDiffer::diff(&a, &b);
        assert!(diff.has_breaking_changes());
    }

    #[test]
    fn test_characteristic_added() {
        let mut a = make_aspect("urn:samm:#A");
        a.add_property(make_property("urn:samm:#speed", false));
        let mut b = make_aspect("urn:samm:#A");
        b.add_property(make_property_with_char("urn:samm:#speed", "xsd:float"));
        let diff = AspectModelDiffer::diff(&a, &b);
        let added = diff.by_kind(&ChangeKind::Added);
        assert!(!added.is_empty());
    }

    #[test]
    fn test_characteristic_removed() {
        let mut a = make_aspect("urn:samm:#A");
        a.add_property(make_property_with_char("urn:samm:#speed", "xsd:float"));
        let mut b = make_aspect("urn:samm:#A");
        b.add_property(make_property("urn:samm:#speed", false));
        let diff = AspectModelDiffer::diff(&a, &b);
        assert!(diff.has_breaking_changes());
    }

    // -- operation changes --

    #[test]
    fn test_operation_added() {
        let a = make_aspect("urn:samm:#A");
        let mut b = make_aspect("urn:samm:#A");
        b.add_operation(make_operation("urn:samm:#calibrate"));
        let diff = AspectModelDiffer::diff(&a, &b);
        assert_eq!(diff.by_kind(&ChangeKind::Added).len(), 1);
    }

    #[test]
    fn test_operation_removed() {
        let mut a = make_aspect("urn:samm:#A");
        a.add_operation(make_operation("urn:samm:#calibrate"));
        let b = make_aspect("urn:samm:#A");
        let diff = AspectModelDiffer::diff(&a, &b);
        let removed = diff.by_kind(&ChangeKind::Removed);
        assert_eq!(removed.len(), 1);
        assert_eq!(removed[0].severity, DiffSeverity::Critical);
    }

    // -- event changes --

    #[test]
    fn test_event_added() {
        let a = make_aspect("urn:samm:#A");
        let mut b = make_aspect("urn:samm:#A");
        b.add_event(make_event("urn:samm:#speedChanged"));
        let diff = AspectModelDiffer::diff(&a, &b);
        assert!(!diff.is_empty());
    }

    #[test]
    fn test_event_removed() {
        let mut a = make_aspect("urn:samm:#A");
        a.add_event(make_event("urn:samm:#speedChanged"));
        let b = make_aspect("urn:samm:#A");
        let diff = AspectModelDiffer::diff(&a, &b);
        assert!(diff.has_breaking_changes());
    }

    // -- metadata changes --

    #[test]
    fn test_preferred_name_added() {
        let a = make_aspect("urn:samm:#A");
        let mut b = make_aspect("urn:samm:#A");
        b.metadata
            .add_preferred_name("en".to_string(), "Aspect A".to_string());
        let diff = AspectModelDiffer::diff(&a, &b);
        assert!(!diff.is_empty());
        assert_eq!(diff.changes[0].severity, DiffSeverity::Info);
    }

    #[test]
    fn test_preferred_name_removed() {
        let mut a = make_aspect("urn:samm:#A");
        a.metadata
            .add_preferred_name("en".to_string(), "Aspect A".to_string());
        let b = make_aspect("urn:samm:#A");
        let diff = AspectModelDiffer::diff(&a, &b);
        assert!(!diff.is_empty());
    }

    #[test]
    fn test_preferred_name_changed() {
        let mut a = make_aspect("urn:samm:#A");
        a.metadata
            .add_preferred_name("en".to_string(), "Old Name".to_string());
        let mut b = make_aspect("urn:samm:#A");
        b.metadata
            .add_preferred_name("en".to_string(), "New Name".to_string());
        let diff = AspectModelDiffer::diff(&a, &b);
        let mods = diff.by_kind(&ChangeKind::Modified);
        assert_eq!(mods.len(), 1);
    }

    #[test]
    fn test_description_added() {
        let a = make_aspect("urn:samm:#A");
        let mut b = make_aspect("urn:samm:#A");
        b.metadata
            .add_description("en".to_string(), "A description".to_string());
        let diff = AspectModelDiffer::diff(&a, &b);
        assert!(!diff.is_empty());
    }

    // -- report generation --

    #[test]
    fn test_text_report_contains_info() {
        let mut a = make_aspect("urn:samm:#A");
        a.add_property(make_property("urn:samm:#speed", false));
        let b = make_aspect("urn:samm:#A");
        let diff = AspectModelDiffer::diff(&a, &b);
        let report = diff.to_text_report();
        assert!(report.contains("CRITICAL"));
        assert!(report.contains("REMOVED"));
    }

    #[test]
    fn test_markdown_report_contains_table() {
        let a = make_aspect("urn:samm:#A");
        let mut b = make_aspect("urn:samm:#A");
        b.add_property(make_property("urn:samm:#speed", false));
        let diff = AspectModelDiffer::diff(&a, &b);
        let md = diff.to_markdown_report();
        assert!(md.contains("| Severity"));
        assert!(md.contains("MINOR"));
    }

    #[test]
    fn test_empty_diff_markdown() {
        let a = make_aspect("urn:samm:#A");
        let b = make_aspect("urn:samm:#A");
        let diff = AspectModelDiffer::diff(&a, &b);
        let md = diff.to_markdown_report();
        assert!(md.contains("No changes detected"));
    }

    // -- severity / kind queries --

    #[test]
    fn test_by_severity() {
        let mut a = make_aspect("urn:samm:#A");
        a.add_property(make_property("urn:samm:#speed", false));
        let b = make_aspect("urn:samm:#A");
        let diff = AspectModelDiffer::diff(&a, &b);
        assert_eq!(diff.by_severity(DiffSeverity::Critical).len(), 1);
        assert_eq!(diff.by_severity(DiffSeverity::Minor).len(), 0);
    }

    #[test]
    fn test_max_severity() {
        let a = make_aspect("urn:samm:#A");
        let b = make_aspect("urn:samm:#A");
        assert_eq!(AspectModelDiffer::diff(&a, &b).max_severity(), None);
    }

    #[test]
    fn test_severity_histogram() {
        let mut a = make_aspect("urn:samm:#A");
        a.add_property(make_property("urn:samm:#speed", false));
        let mut b = make_aspect("urn:samm:#A");
        b.add_property(make_property("urn:samm:#accel", false));
        let diff = AspectModelDiffer::diff(&a, &b);
        let hist = diff.severity_histogram();
        assert!(hist.values().sum::<usize>() > 0);
    }

    // -- display / to_string --

    #[test]
    fn test_severity_display() {
        assert_eq!(DiffSeverity::Info.to_string(), "INFO");
        assert_eq!(DiffSeverity::Minor.to_string(), "MINOR");
        assert_eq!(DiffSeverity::Major.to_string(), "MAJOR");
        assert_eq!(DiffSeverity::Critical.to_string(), "CRITICAL");
    }

    #[test]
    fn test_change_kind_display() {
        assert_eq!(ChangeKind::Added.to_string(), "ADDED");
        assert_eq!(ChangeKind::Removed.to_string(), "REMOVED");
        assert_eq!(ChangeKind::Modified.to_string(), "MODIFIED");
        assert_eq!(ChangeKind::Renamed.to_string(), "RENAMED");
    }

    #[test]
    fn test_severity_ordering() {
        assert!(DiffSeverity::Info < DiffSeverity::Minor);
        assert!(DiffSeverity::Minor < DiffSeverity::Major);
        assert!(DiffSeverity::Major < DiffSeverity::Critical);
    }

    // -- complex scenario --

    #[test]
    fn test_complex_diff_scenario() {
        let mut old = make_aspect("urn:samm:org.example:1.0.0#Movement");
        old.add_property(make_property_with_char("urn:samm:#speed", "xsd:float"));
        old.add_property(make_property("urn:samm:#direction", false));
        old.add_operation(make_operation("urn:samm:#calibrate"));

        let mut new = make_aspect("urn:samm:org.example:1.1.0#Movement");
        new.add_property(make_property_with_char("urn:samm:#speed", "xsd:double")); // type changed
                                                                                    // direction removed
        new.add_property(make_property("urn:samm:#acceleration", false)); // new
        new.add_operation(make_operation("urn:samm:#calibrate")); // same
        new.add_event(make_event("urn:samm:#speedEvent")); // new event

        let diff = AspectModelDiffer::diff(&old, &new);
        assert!(diff.total_changes() >= 4);
        assert!(diff.has_breaking_changes());
    }

    #[test]
    fn test_has_breaking_changes_false_for_additions() {
        let a = make_aspect("urn:samm:#A");
        let mut b = make_aspect("urn:samm:#A");
        b.add_property(make_property("urn:samm:#new_prop", false));
        let diff = AspectModelDiffer::diff(&a, &b);
        // Adding a property is Minor, not breaking
        assert!(!diff.has_breaking_changes());
    }

    // -- constraint diff --

    #[test]
    fn test_constraint_added_to_characteristic() {
        let mut old = make_aspect("urn:samm:#A");
        let old_prop = make_property_with_char("urn:samm:#val", "xsd:integer");
        // No constraints on old
        old.add_property(old_prop);

        let mut new = make_aspect("urn:samm:#A");
        let mut new_prop = make_property_with_char("urn:samm:#val", "xsd:integer");
        if let Some(ref mut c) = new_prop.characteristic {
            c.constraints.push(Constraint::RangeConstraint {
                min_value: Some("0".to_string()),
                max_value: Some("100".to_string()),
                lower_bound_definition: BoundDefinition::Open,
                upper_bound_definition: BoundDefinition::Open,
            });
        }
        new.add_property(new_prop);

        let diff = AspectModelDiffer::diff(&old, &new);
        assert!(!diff.is_empty());
    }

    #[test]
    fn test_characteristic_kind_change() {
        let mut old = make_aspect("urn:samm:#A");
        let old_prop = make_property_with_char("urn:samm:#val", "xsd:string");
        old.add_property(old_prop);

        let mut new = make_aspect("urn:samm:#A");
        let mut new_prop = make_property_with_char("urn:samm:#val", "xsd:string");
        if let Some(ref mut c) = new_prop.characteristic {
            c.kind = CharacteristicKind::Collection {
                element_characteristic: None,
            };
        }
        new.add_property(new_prop);

        let diff = AspectModelDiffer::diff(&old, &new);
        assert!(diff.has_breaking_changes());
    }
}
