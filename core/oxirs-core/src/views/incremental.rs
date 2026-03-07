//! Incremental view maintenance with delta propagation.
//!
//! This module provides:
//! - [`ViewDefinition`]: Declares a named, materialized SPARQL view.
//! - [`DeltaChange`]: Represents a single triple insert or delete.
//! - [`ViewRow`]: One result row in a materialized view (variable → value map).
//! - [`MaterializedView`]: The cached result set of a view together with staleness metadata.
//! - [`IncrementalViewMaintainer`]: Manages a registry of views and propagates delta changes.
//! - [`ViewStalenessDetector`]: Utility helpers to test whether views have grown stale over time.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// ViewDefinition
// ---------------------------------------------------------------------------

/// Metadata that describes a named, materialized SPARQL view.
///
/// The `dependencies` field lists the predicate IRIs that the SPARQL query
/// accesses.  It is used for selective delta propagation: only views that
/// mention a changed predicate in their `dependencies` (or have an empty list,
/// meaning "depends on everything") are invalidated when a triple changes.
#[derive(Debug, Clone)]
pub struct ViewDefinition {
    /// Human-readable name that serves as the unique key in a maintainer.
    pub name: String,
    /// The SPARQL SELECT query that defines the view's result set.
    pub sparql_query: String,
    /// Whether the view caches its result rows (materialized) or is virtual.
    pub is_materialized: bool,
    /// Predicate IRIs this view's query accesses.
    ///
    /// An *empty* list means the view depends on **all** predicates (universal
    /// dependency), so it is invalidated by any triple change.
    pub dependencies: Vec<String>,
}

impl ViewDefinition {
    /// Create a new view definition.
    pub fn new(
        name: impl Into<String>,
        sparql_query: impl Into<String>,
        is_materialized: bool,
        dependencies: Vec<String>,
    ) -> Self {
        Self {
            name: name.into(),
            sparql_query: sparql_query.into(),
            is_materialized,
            dependencies,
        }
    }

    /// Return `true` if the given `predicate` IRI is listed in the
    /// `dependencies`, or if `dependencies` is empty (universal dependency).
    pub fn depends_on(&self, predicate: &str) -> bool {
        self.dependencies.is_empty() || self.dependencies.iter().any(|p| p.as_str() == predicate)
    }
}

// ---------------------------------------------------------------------------
// DeltaChange
// ---------------------------------------------------------------------------

/// A single modification to the underlying triple store.
///
/// Both variants carry the `(subject, predicate, object)` triple as owned
/// `String`s to keep the API self-contained.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeltaChange {
    /// A triple was added to the store.
    Insert {
        subject: String,
        predicate: String,
        object: String,
    },
    /// A triple was removed from the store.
    Delete {
        subject: String,
        predicate: String,
        object: String,
    },
}

impl DeltaChange {
    /// Return the predicate IRI of this change.
    pub fn predicate(&self) -> &str {
        match self {
            DeltaChange::Insert { predicate, .. } | DeltaChange::Delete { predicate, .. } => {
                predicate.as_str()
            }
        }
    }

    /// Return the subject IRI of this change.
    pub fn subject(&self) -> &str {
        match self {
            DeltaChange::Insert { subject, .. } | DeltaChange::Delete { subject, .. } => {
                subject.as_str()
            }
        }
    }

    /// Return the object value of this change.
    pub fn object(&self) -> &str {
        match self {
            DeltaChange::Insert { object, .. } | DeltaChange::Delete { object, .. } => {
                object.as_str()
            }
        }
    }

    /// Return `true` if this is an insertion.
    pub fn is_insert(&self) -> bool {
        matches!(self, DeltaChange::Insert { .. })
    }
}

// ---------------------------------------------------------------------------
// ViewRow
// ---------------------------------------------------------------------------

/// One result row in a materialized view: a mapping from SPARQL variable
/// names to their bound string values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViewRow(pub HashMap<String, String>);

impl ViewRow {
    /// Create a new row from a plain `HashMap`.
    pub fn new(map: HashMap<String, String>) -> Self {
        Self(map)
    }

    /// Convenience constructor for tests and small code paths.
    pub fn from_pairs(pairs: &[(&str, &str)]) -> Self {
        Self(
            pairs
                .iter()
                .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
                .collect(),
        )
    }

    /// Return the value bound to `variable`, if any.
    pub fn get(&self, variable: &str) -> Option<&str> {
        self.0.get(variable).map(|s| s.as_str())
    }
}

// ---------------------------------------------------------------------------
// MaterializedView
// ---------------------------------------------------------------------------

/// The cached result set of a named SPARQL view.
///
/// `last_updated_ms` is a Unix timestamp in milliseconds; `version` is
/// incremented every time the view is refreshed.
pub struct MaterializedView {
    /// The definition that drives this view.
    pub definition: ViewDefinition,
    /// Cached result rows.
    pub rows: Vec<ViewRow>,
    /// Unix timestamp (ms) when the view was last refreshed.
    pub last_updated_ms: i64,
    /// Monotonically increasing refresh counter.
    pub version: u64,
    /// Whether the cached rows are out-of-date.
    pub is_stale: bool,
}

impl MaterializedView {
    /// Create a new `MaterializedView` with the given initial rows.
    ///
    /// `last_updated_ms` is set to the current wall-clock time and `version`
    /// starts at 0.
    pub fn new(definition: ViewDefinition, initial_rows: Vec<ViewRow>) -> Self {
        Self {
            definition,
            rows: initial_rows,
            last_updated_ms: now_ms(),
            version: 0,
            is_stale: false,
        }
    }

    /// Replace the current rows with `new_rows`, clear the stale flag, and
    /// bump `version`.
    pub fn refresh(&mut self, new_rows: Vec<ViewRow>) {
        self.rows = new_rows;
        self.last_updated_ms = now_ms();
        self.version += 1;
        self.is_stale = false;
    }

    /// Mark the view as stale (i.e. its cached rows need to be recomputed).
    pub fn invalidate(&mut self) {
        self.is_stale = true;
    }
}

// ---------------------------------------------------------------------------
// IncrementalViewMaintainer
// ---------------------------------------------------------------------------

/// Manages a collection of [`MaterializedView`]s and propagates delta changes.
///
/// # Workflow
///
/// 1. Call [`register_view`] to create a new view from a [`ViewDefinition`] and
///    an initial result set.
/// 2. Call [`apply_delta`] (or [`queue_change`] + [`flush_changes`]) whenever
///    the underlying triple store changes.
/// 3. Invalidated views can be re-evaluated by the caller and refreshed with
///    new rows via internal access to the views map (or by replacing the whole
///    view).
///
/// [`register_view`]: IncrementalViewMaintainer::register_view
/// [`apply_delta`]: IncrementalViewMaintainer::apply_delta
/// [`queue_change`]: IncrementalViewMaintainer::queue_change
/// [`flush_changes`]: IncrementalViewMaintainer::flush_changes
pub struct IncrementalViewMaintainer {
    views: HashMap<String, MaterializedView>,
    change_queue: Vec<DeltaChange>,
}

impl IncrementalViewMaintainer {
    /// Create an empty maintainer.
    pub fn new() -> Self {
        Self {
            views: HashMap::new(),
            change_queue: Vec::new(),
        }
    }

    /// Register a named view.
    ///
    /// If a view with the same `definition.name` already exists it is
    /// replaced.
    pub fn register_view(&mut self, def: ViewDefinition, initial_rows: Vec<ViewRow>) {
        let name = def.name.clone();
        let view = MaterializedView::new(def, initial_rows);
        self.views.insert(name, view);
    }

    /// Apply a single [`DeltaChange`] immediately.
    ///
    /// Returns the names of views that were invalidated (i.e. whose
    /// `dependencies` include the changed predicate or are empty).
    /// Views that are already stale are *not* returned again (no-duplicate
    /// semantics).
    pub fn apply_delta(&mut self, change: DeltaChange) -> Vec<String> {
        let predicate = change.predicate().to_string();
        let mut invalidated = Vec::new();

        for (name, view) in self.views.iter_mut() {
            if !view.is_stale && view.definition.depends_on(&predicate) {
                view.is_stale = true;
                invalidated.push(name.clone());
            }
        }

        invalidated
    }

    /// Mark a view as stale by name.
    ///
    /// Does nothing if the view does not exist.
    pub fn invalidate_view(&mut self, name: &str) {
        if let Some(view) = self.views.get_mut(name) {
            view.invalidate();
        }
    }

    /// Return an immutable reference to a view.
    pub fn get_view(&self, name: &str) -> Option<&MaterializedView> {
        self.views.get(name)
    }

    /// Return the names of all registered views, in unspecified order.
    pub fn list_views(&self) -> Vec<&str> {
        self.views.keys().map(|s| s.as_str()).collect()
    }

    /// Return the names of all views that depend on `predicate`.
    ///
    /// This includes views with an empty `dependencies` list (universal
    /// dependency).
    pub fn affected_views(&self, predicate: &str) -> Vec<&str> {
        self.views
            .iter()
            .filter(|(_, v)| v.definition.depends_on(predicate))
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Push a change onto the internal queue for later batch processing.
    pub fn queue_change(&mut self, change: DeltaChange) {
        self.change_queue.push(change);
    }

    /// Apply all queued changes at once and clear the queue.
    ///
    /// Returns a map of `view_name → rows_changed`.  Because this
    /// implementation uses a mark-stale strategy, invalidated views are
    /// reported with `0` changed rows.  Views that were not affected are not
    /// included in the returned map.
    pub fn flush_changes(&mut self) -> HashMap<String, usize> {
        let changes: Vec<DeltaChange> = self.change_queue.drain(..).collect();
        let mut result: HashMap<String, usize> = HashMap::new();

        for change in changes {
            let invalidated = self.apply_delta(change);
            for name in invalidated {
                // Mark-stale strategy: report 0 rows changed for each newly
                // invalidated view.  Views already stale are not re-added.
                result.entry(name).or_insert(0);
            }
        }

        result
    }

    /// Return the number of registered views.
    pub fn view_count(&self) -> usize {
        self.views.len()
    }

    /// Return the total number of cached rows across all non-stale views.
    pub fn total_rows(&self) -> usize {
        self.views
            .values()
            .filter(|v| !v.is_stale)
            .map(|v| v.rows.len())
            .sum()
    }
}

impl Default for IncrementalViewMaintainer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ViewStalenessDetector
// ---------------------------------------------------------------------------

/// Utility helpers for determining whether materialized views have grown stale
/// based on wall-clock age.
pub struct ViewStalenessDetector;

impl ViewStalenessDetector {
    /// Return `true` if `view` was last updated more than `max_age_ms`
    /// milliseconds ago, or if it is already marked stale.
    pub fn is_stale(view: &MaterializedView, max_age_ms: i64) -> bool {
        if view.is_stale {
            return true;
        }
        let age = now_ms() - view.last_updated_ms;
        age > max_age_ms
    }

    /// Filter a slice of view references and return those that are stale
    /// according to [`is_stale`].
    ///
    /// [`is_stale`]: ViewStalenessDetector::is_stale
    pub fn stale_views<'a>(
        views: &[&'a MaterializedView],
        max_age_ms: i64,
    ) -> Vec<&'a MaterializedView> {
        views
            .iter()
            .copied()
            .filter(|v| Self::is_stale(v, max_age_ms))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    // ---- helpers ----

    fn make_def(name: &str, deps: &[&str], materialized: bool) -> ViewDefinition {
        ViewDefinition::new(
            name,
            format!("SELECT * WHERE {{ ?s <http://p/{}> ?o }}", name),
            materialized,
            deps.iter().map(|s| s.to_string()).collect(),
        )
    }

    fn rows(n: usize) -> Vec<ViewRow> {
        (0..n)
            .map(|i| ViewRow::from_pairs(&[("s", &format!("s{}", i)), ("o", &format!("o{}", i))]))
            .collect()
    }

    // ---- ViewDefinition ----

    #[test]
    fn test_view_definition_depends_on_listed_predicate() {
        let def = make_def("v", &["http://p/age"], true);
        assert!(def.depends_on("http://p/age"));
        assert!(!def.depends_on("http://p/name"));
    }

    #[test]
    fn test_view_definition_empty_deps_matches_all() {
        let def = make_def("v", &[], true);
        assert!(def.depends_on("http://anything"));
    }

    #[test]
    fn test_view_definition_multiple_deps() {
        let def = make_def("v", &["http://p/age", "http://p/name"], false);
        assert!(def.depends_on("http://p/age"));
        assert!(def.depends_on("http://p/name"));
        assert!(!def.depends_on("http://p/color"));
    }

    // ---- DeltaChange ----

    #[test]
    fn test_delta_change_insert_accessors() {
        let d = DeltaChange::Insert {
            subject: "s".into(),
            predicate: "p".into(),
            object: "o".into(),
        };
        assert_eq!(d.subject(), "s");
        assert_eq!(d.predicate(), "p");
        assert_eq!(d.object(), "o");
        assert!(d.is_insert());
    }

    #[test]
    fn test_delta_change_delete_accessors() {
        let d = DeltaChange::Delete {
            subject: "s".into(),
            predicate: "p".into(),
            object: "o".into(),
        };
        assert!(!d.is_insert());
    }

    #[test]
    fn test_delta_change_equality() {
        let a = DeltaChange::Insert {
            subject: "s".into(),
            predicate: "p".into(),
            object: "o".into(),
        };
        let b = a.clone();
        assert_eq!(a, b);
    }

    // ---- ViewRow ----

    #[test]
    fn test_view_row_get() {
        let row = ViewRow::from_pairs(&[("x", "Alice"), ("y", "30")]);
        assert_eq!(row.get("x"), Some("Alice"));
        assert_eq!(row.get("z"), None);
    }

    // ---- MaterializedView ----

    #[test]
    fn test_materialized_view_initial_state() {
        let def = make_def("v", &["http://p/age"], true);
        let mv = MaterializedView::new(def, rows(3));
        assert_eq!(mv.rows.len(), 3);
        assert_eq!(mv.version, 0);
        assert!(!mv.is_stale);
    }

    #[test]
    fn test_materialized_view_refresh_bumps_version() {
        let def = make_def("v", &["http://p/age"], true);
        let mut mv = MaterializedView::new(def, rows(2));
        mv.refresh(rows(5));
        assert_eq!(mv.rows.len(), 5);
        assert_eq!(mv.version, 1);
        assert!(!mv.is_stale);
    }

    #[test]
    fn test_materialized_view_invalidate_sets_stale() {
        let def = make_def("v", &["http://p/age"], true);
        let mut mv = MaterializedView::new(def, rows(2));
        mv.invalidate();
        assert!(mv.is_stale);
    }

    // ---- IncrementalViewMaintainer ----

    #[test]
    fn test_maintainer_register_and_count() {
        let mut m = IncrementalViewMaintainer::new();
        m.register_view(make_def("v1", &["http://p/age"], true), rows(1));
        m.register_view(make_def("v2", &["http://p/name"], true), rows(2));
        assert_eq!(m.view_count(), 2);
    }

    #[test]
    fn test_maintainer_apply_delta_invalidates_affected() {
        let mut m = IncrementalViewMaintainer::new();
        m.register_view(make_def("v_age", &["http://p/age"], true), rows(2));
        m.register_view(make_def("v_name", &["http://p/name"], true), rows(3));

        let changed = m.apply_delta(DeltaChange::Insert {
            subject: "s".into(),
            predicate: "http://p/age".into(),
            object: "42".into(),
        });

        assert_eq!(changed.len(), 1);
        assert_eq!(changed[0], "v_age");
        assert!(m.get_view("v_age").map(|v| v.is_stale).unwrap_or(false));
        assert!(!m.get_view("v_name").map(|v| v.is_stale).unwrap_or(true));
    }

    #[test]
    fn test_maintainer_apply_delta_already_stale_not_returned_again() {
        let mut m = IncrementalViewMaintainer::new();
        m.register_view(make_def("v", &["http://p/age"], true), rows(1));

        m.apply_delta(DeltaChange::Insert {
            subject: "s".into(),
            predicate: "http://p/age".into(),
            object: "1".into(),
        });
        // Second delta on the same predicate — v is already stale.
        let changed = m.apply_delta(DeltaChange::Insert {
            subject: "s2".into(),
            predicate: "http://p/age".into(),
            object: "2".into(),
        });
        assert!(
            changed.is_empty(),
            "Already stale, should not be re-reported"
        );
    }

    #[test]
    fn test_maintainer_apply_delta_universal_dependency() {
        let mut m = IncrementalViewMaintainer::new();
        m.register_view(make_def("v_all", &[], true), rows(0)); // empty deps = all

        let changed = m.apply_delta(DeltaChange::Delete {
            subject: "s".into(),
            predicate: "http://totally/unknown".into(),
            object: "o".into(),
        });
        assert_eq!(changed.len(), 1);
    }

    #[test]
    fn test_maintainer_invalidate_view_by_name() {
        let mut m = IncrementalViewMaintainer::new();
        m.register_view(make_def("v", &["http://p/x"], true), rows(1));
        m.invalidate_view("v");
        assert!(m.get_view("v").map(|v| v.is_stale).unwrap_or(false));
    }

    #[test]
    fn test_maintainer_invalidate_nonexistent_view_is_noop() {
        let mut m = IncrementalViewMaintainer::new();
        // Should not panic.
        m.invalidate_view("does_not_exist");
    }

    #[test]
    fn test_maintainer_affected_views_returns_matching() {
        let mut m = IncrementalViewMaintainer::new();
        m.register_view(make_def("v_age", &["http://p/age"], true), rows(0));
        m.register_view(make_def("v_name", &["http://p/name"], true), rows(0));
        m.register_view(make_def("v_all", &[], true), rows(0));

        let mut affected = m.affected_views("http://p/age");
        affected.sort_unstable();
        // v_age and v_all should be returned; v_name should not.
        assert!(affected.contains(&"v_age"));
        assert!(affected.contains(&"v_all"));
        assert!(!affected.contains(&"v_name"));
    }

    #[test]
    fn test_maintainer_queue_and_flush_changes() {
        let mut m = IncrementalViewMaintainer::new();
        m.register_view(make_def("v_age", &["http://p/age"], true), rows(3));
        m.register_view(make_def("v_name", &["http://p/name"], true), rows(2));

        m.queue_change(DeltaChange::Insert {
            subject: "s1".into(),
            predicate: "http://p/age".into(),
            object: "25".into(),
        });
        m.queue_change(DeltaChange::Delete {
            subject: "s2".into(),
            predicate: "http://p/name".into(),
            object: "Alice".into(),
        });

        let result = m.flush_changes();
        assert_eq!(result.len(), 2);
        assert!(result.contains_key("v_age"));
        assert!(result.contains_key("v_name"));
        // Mark-stale strategy: rows_changed reported as 0.
        assert_eq!(result["v_age"], 0);
        assert_eq!(result["v_name"], 0);
    }

    #[test]
    fn test_maintainer_flush_clears_queue() {
        let mut m = IncrementalViewMaintainer::new();
        m.register_view(make_def("v", &["http://p/x"], true), rows(1));
        m.queue_change(DeltaChange::Insert {
            subject: "s".into(),
            predicate: "http://p/x".into(),
            object: "o".into(),
        });
        m.flush_changes();
        // Flushing again with empty queue returns empty map.
        let second = m.flush_changes();
        assert!(second.is_empty());
    }

    #[test]
    fn test_maintainer_total_rows_excludes_stale() {
        let mut m = IncrementalViewMaintainer::new();
        m.register_view(make_def("v1", &["http://p/age"], true), rows(5));
        m.register_view(make_def("v2", &["http://p/name"], true), rows(3));

        assert_eq!(m.total_rows(), 8);

        // Invalidate v1.
        m.apply_delta(DeltaChange::Insert {
            subject: "s".into(),
            predicate: "http://p/age".into(),
            object: "1".into(),
        });

        assert_eq!(m.total_rows(), 3); // v1 is stale, only v2 counts.
    }

    #[test]
    fn test_maintainer_list_views() {
        let mut m = IncrementalViewMaintainer::new();
        m.register_view(make_def("alpha", &[], true), rows(0));
        m.register_view(make_def("beta", &[], true), rows(0));

        let mut names = m.list_views();
        names.sort_unstable();
        assert_eq!(names, vec!["alpha", "beta"]);
    }

    #[test]
    fn test_maintainer_replace_existing_view() {
        let mut m = IncrementalViewMaintainer::new();
        m.register_view(make_def("v", &["http://p/x"], true), rows(2));
        // Re-register with the same name but different rows.
        m.register_view(make_def("v", &["http://p/x"], true), rows(10));
        assert_eq!(m.view_count(), 1);
        assert_eq!(m.get_view("v").map(|v| v.rows.len()), Some(10));
    }

    // ---- ViewStalenessDetector ----

    #[test]
    fn test_staleness_detector_explicitly_stale() {
        let def = make_def("v", &[], true);
        let mut mv = MaterializedView::new(def, rows(1));
        mv.is_stale = true;
        // Any max_age_ms → still stale.
        assert!(ViewStalenessDetector::is_stale(&mv, i64::MAX));
    }

    #[test]
    fn test_staleness_detector_freshly_created_not_stale() {
        let def = make_def("v", &[], true);
        let mv = MaterializedView::new(def, rows(1));
        // 1 hour should be well within freshness.
        assert!(!ViewStalenessDetector::is_stale(&mv, 3_600_000));
    }

    #[test]
    fn test_staleness_detector_zero_max_age_always_stale() {
        let def = make_def("v", &[], true);
        // Sleep briefly so that now_ms() - last_updated_ms > 0.
        let mv = MaterializedView::new(def, rows(0));
        thread::sleep(Duration::from_millis(5));
        assert!(ViewStalenessDetector::is_stale(&mv, 0));
    }

    #[test]
    fn test_staleness_detector_stale_views_filters_correctly() {
        let def1 = make_def("v1", &[], true);
        let def2 = make_def("v2", &[], true);
        let mv1 = MaterializedView::new(def1, rows(0));
        let mut mv2 = MaterializedView::new(def2, rows(0));
        mv2.is_stale = true;

        let stale = ViewStalenessDetector::stale_views(&[&mv1, &mv2], 3_600_000);
        assert_eq!(stale.len(), 1);
        assert!(stale[0].is_stale);
    }
}
