//! Incremental view maintenance for materialized SPARQL views.
//!
//! This module provides machinery for keeping materialized views up-to-date as triples are
//! inserted into or deleted from the dataset, without requiring a full re-computation of the
//! underlying SPARQL query for every change.
//!
//! # Architecture
//!
//! The key insight of incremental view maintenance (IVM) is that most triple changes only affect
//! a small fraction of a materialized view.  The system works as follows:
//!
//! 1. A [`ViewDefinition`] captures the SPARQL query and the predicates it accesses.
//! 2. When triples change, the caller publishes a list of [`TripleDelta`] events to the
//!    [`ViewManager`].
//! 3. The manager checks each registered view's `accessed_predicates` against the deltas.
//! 4. Views that are affected are marked **stale**; the caller is responsible for refreshing them
//!    (by re-executing the query and calling [`ViewManager::refresh_view`]).
//!
//! # Staleness vs Re-computation
//!
//! Full incremental re-computation (propagating delta rows through relational operators) is
//! extremely complex to implement correctly for all SPARQL constructs, especially OPTIONAL, MINUS,
//! and aggregations.  This implementation therefore uses a **mark-stale** strategy: affected views
//! are invalidated and the caller must re-run the SPARQL query.  This is still far cheaper than
//! always re-running all queries because:
//!
//! - Only views that *actually* overlap with the changed predicates are re-evaluated.
//! - Views with no overlap remain fully valid and can serve cached results.
//!
//! Future versions may add true delta propagation for simple BGP-only views.

use std::collections::HashMap;
use std::time::Instant;

/// A change to the triple store.
///
/// `Insert` means a new triple `(subject, predicate, object)` was added.
/// `Delete` means an existing triple was removed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TripleDelta {
    /// A triple was inserted.
    Insert(String, String, String),
    /// A triple was deleted.
    Delete(String, String, String),
}

impl TripleDelta {
    /// Return the predicate IRI of this delta.
    pub fn predicate(&self) -> &str {
        match self {
            TripleDelta::Insert(_, p, _) | TripleDelta::Delete(_, p, _) => p.as_str(),
        }
    }

    /// Return the subject IRI of this delta.
    pub fn subject(&self) -> &str {
        match self {
            TripleDelta::Insert(s, _, _) | TripleDelta::Delete(s, _, _) => s.as_str(),
        }
    }

    /// Return the object value of this delta.
    pub fn object(&self) -> &str {
        match self {
            TripleDelta::Insert(_, _, o) | TripleDelta::Delete(_, _, o) => o.as_str(),
        }
    }

    /// Return `true` if this is an insertion.
    pub fn is_insert(&self) -> bool {
        matches!(self, TripleDelta::Insert(_, _, _))
    }
}

/// Definition of a materialized view.
///
/// A view is identified by a unique `id` and backed by a SPARQL SELECT query.  The
/// `accessed_predicates` list is used to decide which triple deltas can affect this view; if the
/// list is empty the view is considered to be affected by **all** changes.
#[derive(Debug, Clone)]
pub struct ViewDefinition {
    /// Unique identifier for this view (e.g. a UUID or user-supplied name).
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// SPARQL SELECT query whose result set defines this view.
    pub sparql_query: String,
    /// Predicate IRIs that this view's query reads.  Used for selective delta propagation.
    /// If empty, the view is assumed to depend on all predicates.
    pub accessed_predicates: Vec<String>,
    /// When this definition was registered.
    pub created_at: Instant,
}

impl ViewDefinition {
    /// Create a new view definition.
    ///
    /// `accessed_predicates` should list every predicate IRI that appears in the SPARQL query's
    /// triple patterns.  An empty list means "depends on everything."
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        sparql_query: impl Into<String>,
        accessed_predicates: Vec<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            sparql_query: sparql_query.into(),
            accessed_predicates,
            created_at: Instant::now(),
        }
    }
}

/// The current materialized state of a SPARQL view.
///
/// Each row is a variable binding map `{variable_name → value}`.
pub struct MaterializedView {
    /// Definition driving this view.
    pub definition: ViewDefinition,
    /// Cached result rows.
    pub rows: Vec<HashMap<String, String>>,
    /// Snapshot of `rows.len()` for quick access.
    pub row_count: usize,
    /// When the view data was last refreshed.
    pub last_updated: Instant,
    /// Cumulative number of times this view has been refreshed.
    pub update_count: u64,
    /// Whether the view needs to be re-evaluated before it can be used.
    pub is_stale: bool,
}

impl MaterializedView {
    /// Construct a new `MaterializedView` from a definition and an initial result set.
    pub fn new(definition: ViewDefinition, initial_rows: Vec<HashMap<String, String>>) -> Self {
        let row_count = initial_rows.len();
        Self {
            definition,
            rows: initial_rows,
            row_count,
            last_updated: Instant::now(),
            update_count: 0,
            is_stale: false,
        }
    }

    /// Apply a batch of deltas, marking this view stale if any delta affects it.
    ///
    /// Returns `true` if the view was newly marked stale by this call.
    pub fn apply_deltas(&mut self, deltas: &[TripleDelta]) -> bool {
        if self.is_stale {
            // Already stale — no need to re-check.
            return false;
        }
        for delta in deltas {
            if self.is_affected_by(delta) {
                self.is_stale = true;
                return true;
            }
        }
        false
    }

    /// Return `true` if `delta` could affect this view's result set.
    ///
    /// A view is considered affected when:
    /// - Its `accessed_predicates` list is empty (depends on everything), **or**
    /// - The delta's predicate appears in `accessed_predicates`.
    pub fn is_affected_by(&self, delta: &TripleDelta) -> bool {
        if self.definition.accessed_predicates.is_empty() {
            return true;
        }
        let pred = delta.predicate();
        self.definition
            .accessed_predicates
            .iter()
            .any(|p| p.as_str() == pred)
    }

    /// Replace the cached rows with `new_rows` and clear the stale flag.
    pub fn refresh(&mut self, new_rows: Vec<HashMap<String, String>>) {
        self.row_count = new_rows.len();
        self.rows = new_rows;
        self.last_updated = Instant::now();
        self.update_count += 1;
        self.is_stale = false;
    }
}

/// Manager for a collection of materialized views.
///
/// The manager owns all [`MaterializedView`] instances and handles delta propagation.
pub struct ViewManager {
    views: HashMap<String, MaterializedView>,
}

impl ViewManager {
    /// Create a new, empty view manager.
    pub fn new() -> Self {
        Self {
            views: HashMap::new(),
        }
    }

    /// Register a new view with its initial result rows.
    ///
    /// If a view with the same `id` already exists it is replaced.
    pub fn register_view(
        &mut self,
        definition: ViewDefinition,
        initial_rows: Vec<HashMap<String, String>>,
    ) {
        let id = definition.id.clone();
        let view = MaterializedView::new(definition, initial_rows);
        self.views.insert(id, view);
    }

    /// Remove a view by its ID.
    ///
    /// Returns `true` if the view existed and was removed, `false` otherwise.
    pub fn drop_view(&mut self, view_id: &str) -> bool {
        self.views.remove(view_id).is_some()
    }

    /// Propagate a batch of deltas to all registered views.
    ///
    /// Views whose predicate sets overlap with the deltas are marked stale.
    ///
    /// Returns the IDs of all views that were newly marked stale by this call.
    pub fn propagate_deltas(&mut self, deltas: &[TripleDelta]) -> Vec<String> {
        let mut stale_ids: Vec<String> = Vec::new();
        for (id, view) in self.views.iter_mut() {
            if view.apply_deltas(deltas) {
                stale_ids.push(id.clone());
            }
        }
        stale_ids
    }

    /// Return the current rows of a view, or `None` if the view does not exist or is stale.
    pub fn get_view_data(&self, view_id: &str) -> Option<&[HashMap<String, String>]> {
        let view = self.views.get(view_id)?;
        if view.is_stale {
            None
        } else {
            Some(&view.rows)
        }
    }

    /// Refresh a stale view with freshly-computed result rows.
    ///
    /// This clears the stale flag so the view is immediately queryable again.
    /// Does nothing if the view does not exist.
    pub fn refresh_view(&mut self, view_id: &str, new_rows: Vec<HashMap<String, String>>) {
        if let Some(view) = self.views.get_mut(view_id) {
            view.refresh(new_rows);
        }
    }

    /// Return the IDs of all currently stale views.
    pub fn stale_views(&self) -> Vec<&str> {
        self.views
            .iter()
            .filter(|(_, v)| v.is_stale)
            .map(|(id, _)| id.as_str())
            .collect()
    }

    /// Return the total number of registered views.
    pub fn view_count(&self) -> usize {
        self.views.len()
    }

    /// Return a reference to a view, regardless of stale status.
    pub fn get_view(&self, view_id: &str) -> Option<&MaterializedView> {
        self.views.get(view_id)
    }

    /// Return an iterator over all view IDs.
    pub fn view_ids(&self) -> impl Iterator<Item = &str> {
        self.views.keys().map(|s| s.as_str())
    }
}

impl Default for ViewManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_row(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect()
    }

    fn simple_view(id: &str, predicates: Vec<&str>) -> ViewDefinition {
        ViewDefinition::new(
            id,
            format!("View {}", id),
            format!("SELECT * WHERE {{ ?s <http://p/{}> ?o }}", id),
            predicates.into_iter().map(|s| s.to_string()).collect(),
        )
    }

    // --- TripleDelta tests ---

    #[test]
    fn test_triple_delta_predicate() {
        let d = TripleDelta::Insert("s".into(), "p".into(), "o".into());
        assert_eq!(d.predicate(), "p");
        assert_eq!(d.subject(), "s");
        assert_eq!(d.object(), "o");
        assert!(d.is_insert());
    }

    #[test]
    fn test_triple_delta_delete() {
        let d = TripleDelta::Delete("s".into(), "p".into(), "o".into());
        assert!(!d.is_insert());
    }

    #[test]
    fn test_triple_delta_equality() {
        let a = TripleDelta::Insert("s".into(), "p".into(), "o".into());
        let b = TripleDelta::Insert("s".into(), "p".into(), "o".into());
        let c = TripleDelta::Delete("s".into(), "p".into(), "o".into());
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    // --- ViewDefinition tests ---

    #[test]
    fn test_view_definition_creation() {
        let def = ViewDefinition::new(
            "v1",
            "My View",
            "SELECT ?s WHERE { ?s <http://p/name> ?o }",
            vec!["http://p/name".to_string()],
        );
        assert_eq!(def.id, "v1");
        assert_eq!(def.accessed_predicates.len(), 1);
    }

    // --- MaterializedView tests ---

    #[test]
    fn test_materialized_view_not_stale_initially() {
        let def = simple_view("v1", vec!["http://p/age"]);
        let rows = vec![make_row(&[("s", "Alice"), ("o", "30")])];
        let view = MaterializedView::new(def, rows);
        assert!(!view.is_stale);
        assert_eq!(view.row_count, 1);
    }

    #[test]
    fn test_is_affected_by_matching_predicate() {
        let def = simple_view("v1", vec!["http://p/age"]);
        let view = MaterializedView::new(def, vec![]);

        let delta = TripleDelta::Insert("s".into(), "http://p/age".into(), "25".into());
        assert!(view.is_affected_by(&delta));
    }

    #[test]
    fn test_is_affected_by_non_matching_predicate() {
        let def = simple_view("v1", vec!["http://p/age"]);
        let view = MaterializedView::new(def, vec![]);

        let delta = TripleDelta::Insert("s".into(), "http://p/name".into(), "Alice".into());
        assert!(!view.is_affected_by(&delta));
    }

    #[test]
    fn test_is_affected_by_empty_predicates_matches_all() {
        let def = simple_view("v1", vec![]); // empty = depends on everything
        let view = MaterializedView::new(def, vec![]);

        let delta = TripleDelta::Insert("s".into(), "http://any/predicate".into(), "o".into());
        assert!(view.is_affected_by(&delta));
    }

    #[test]
    fn test_apply_deltas_marks_stale() {
        let def = simple_view("v1", vec!["http://p/age"]);
        let mut view = MaterializedView::new(def, vec![]);

        let deltas = vec![TripleDelta::Insert(
            "Alice".into(),
            "http://p/age".into(),
            "30".into(),
        )];
        let newly_stale = view.apply_deltas(&deltas);
        assert!(newly_stale);
        assert!(view.is_stale);
    }

    #[test]
    fn test_apply_deltas_no_effect_different_predicate() {
        let def = simple_view("v1", vec!["http://p/age"]);
        let mut view = MaterializedView::new(def, vec![]);

        let deltas = vec![TripleDelta::Insert(
            "Alice".into(),
            "http://p/name".into(),
            "Alice".into(),
        )];
        let newly_stale = view.apply_deltas(&deltas);
        assert!(!newly_stale);
        assert!(!view.is_stale);
    }

    #[test]
    fn test_apply_deltas_already_stale_returns_false() {
        let def = simple_view("v1", vec!["http://p/age"]);
        let mut view = MaterializedView::new(def, vec![]);
        view.is_stale = true; // manually set stale

        let deltas = vec![TripleDelta::Insert(
            "Alice".into(),
            "http://p/age".into(),
            "30".into(),
        )];
        let newly_stale = view.apply_deltas(&deltas);
        assert!(
            !newly_stale,
            "Already stale, so apply_deltas should return false"
        );
    }

    #[test]
    fn test_refresh_clears_stale_flag() {
        let def = simple_view("v1", vec!["http://p/age"]);
        let mut view = MaterializedView::new(def, vec![]);
        view.is_stale = true;

        let new_rows = vec![make_row(&[("s", "Bob"), ("o", "42")])];
        view.refresh(new_rows);

        assert!(!view.is_stale);
        assert_eq!(view.row_count, 1);
        assert_eq!(view.update_count, 1);
    }

    // --- ViewManager tests ---

    #[test]
    fn test_view_manager_register_and_count() {
        let mut mgr = ViewManager::new();
        mgr.register_view(simple_view("v1", vec!["http://p/age"]), vec![]);
        mgr.register_view(simple_view("v2", vec!["http://p/name"]), vec![]);
        assert_eq!(mgr.view_count(), 2);
    }

    #[test]
    fn test_view_manager_drop_view() {
        let mut mgr = ViewManager::new();
        mgr.register_view(simple_view("v1", vec!["http://p/age"]), vec![]);
        assert!(mgr.drop_view("v1"));
        assert!(!mgr.drop_view("v1")); // already gone
        assert_eq!(mgr.view_count(), 0);
    }

    #[test]
    fn test_view_manager_propagate_deltas_selective() {
        let mut mgr = ViewManager::new();
        mgr.register_view(simple_view("v_age", vec!["http://p/age"]), vec![]);
        mgr.register_view(simple_view("v_name", vec!["http://p/name"]), vec![]);

        let deltas = vec![TripleDelta::Insert(
            "Alice".into(),
            "http://p/age".into(),
            "30".into(),
        )];
        let stale = mgr.propagate_deltas(&deltas);
        assert_eq!(stale.len(), 1);
        assert_eq!(stale[0], "v_age");

        // v_name should not be stale
        let all_stale = mgr.stale_views();
        assert!(all_stale.contains(&"v_age"));
        assert!(!all_stale.contains(&"v_name"));
    }

    #[test]
    fn test_view_manager_propagate_deltas_wildcard_view() {
        let mut mgr = ViewManager::new();
        // View with no accessed_predicates matches everything
        mgr.register_view(simple_view("v_all", vec![]), vec![]);

        let deltas = vec![TripleDelta::Delete(
            "s".into(),
            "http://any/pred".into(),
            "o".into(),
        )];
        let stale = mgr.propagate_deltas(&deltas);
        assert_eq!(stale.len(), 1);
    }

    #[test]
    fn test_view_manager_get_view_data_not_stale() {
        let mut mgr = ViewManager::new();
        let rows = vec![make_row(&[("s", "Alice")])];
        mgr.register_view(simple_view("v1", vec!["http://p/age"]), rows.clone());

        let data = mgr.get_view_data("v1");
        assert!(data.is_some());
        assert_eq!(data.expect("data should be available").len(), 1);
    }

    #[test]
    fn test_view_manager_get_view_data_stale_returns_none() {
        let mut mgr = ViewManager::new();
        mgr.register_view(simple_view("v1", vec!["http://p/age"]), vec![]);

        // Mark stale
        let deltas = vec![TripleDelta::Insert(
            "s".into(),
            "http://p/age".into(),
            "99".into(),
        )];
        mgr.propagate_deltas(&deltas);

        // get_view_data should return None for stale views
        assert!(mgr.get_view_data("v1").is_none());
    }

    #[test]
    fn test_view_manager_refresh_view() {
        let mut mgr = ViewManager::new();
        mgr.register_view(simple_view("v1", vec!["http://p/age"]), vec![]);

        // Mark stale
        mgr.propagate_deltas(&[TripleDelta::Insert(
            "s".into(),
            "http://p/age".into(),
            "10".into(),
        )]);

        // Refresh with new data
        let new_rows = vec![
            make_row(&[("s", "Alice"), ("o", "10")]),
            make_row(&[("s", "Bob"), ("o", "20")]),
        ];
        mgr.refresh_view("v1", new_rows);

        let data = mgr.get_view_data("v1").expect("should have fresh data");
        assert_eq!(data.len(), 2);
    }

    #[test]
    fn test_view_manager_stale_views_empty_initially() {
        let mut mgr = ViewManager::new();
        mgr.register_view(simple_view("v1", vec!["http://p/age"]), vec![]);
        assert!(mgr.stale_views().is_empty());
    }

    #[test]
    fn test_view_manager_multiple_deltas_one_call() {
        let mut mgr = ViewManager::new();
        mgr.register_view(simple_view("v_age", vec!["http://p/age"]), vec![]);
        mgr.register_view(simple_view("v_name", vec!["http://p/name"]), vec![]);
        mgr.register_view(simple_view("v_color", vec!["http://p/color"]), vec![]);

        let deltas = vec![
            TripleDelta::Insert("s1".into(), "http://p/age".into(), "30".into()),
            TripleDelta::Insert("s2".into(), "http://p/name".into(), "Alice".into()),
        ];
        let stale = mgr.propagate_deltas(&deltas);
        assert_eq!(stale.len(), 2);
        assert!(mgr.stale_views().contains(&"v_age"));
        assert!(mgr.stale_views().contains(&"v_name"));
        assert!(!mgr.stale_views().contains(&"v_color"));
    }

    #[test]
    fn test_view_manager_refresh_nonexistent_view_is_noop() {
        let mut mgr = ViewManager::new();
        // Should not panic or error
        mgr.refresh_view("nonexistent", vec![]);
    }

    #[test]
    fn test_view_definition_empty_predicates_semantics() {
        // A view with no accessed_predicates must be affected by ANY delta.
        let def = ViewDefinition::new("v", "All", "SELECT * WHERE { ?s ?p ?o }", vec![]);
        let view = MaterializedView::new(def, vec![]);
        let d = TripleDelta::Delete("s".into(), "http://totally/random".into(), "o".into());
        assert!(view.is_affected_by(&d));
    }
}
