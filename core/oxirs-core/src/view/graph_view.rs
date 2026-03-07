//! Named graph views for OxiRS.
//!
//! Provides:
//! - [`GraphView`]: Read-only view over a subset of RDF triples.
//! - [`FilteredView`]: View with subject/predicate/object filters applied.
//! - [`UnionView`]: Virtual union over multiple named graph views.
//! - [`MergedView`]: Merges a default graph with one or more named graphs.
//! - [`ViewMaterializer`]: Materializes and caches view data.

use std::collections::HashMap;

/// An RDF triple (subject, predicate, object) — all as string-encoded IRIs/literals.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RdfTriple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

impl RdfTriple {
    pub fn new(s: &str, p: &str, o: &str) -> Self {
        Self {
            subject: s.to_string(),
            predicate: p.to_string(),
            object: o.to_string(),
        }
    }

    /// Returns `true` if `pattern` matches this triple.
    /// `None` in any slot is a wildcard.
    pub fn matches(
        &self,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
    ) -> bool {
        subject.map(|s| s == self.subject).unwrap_or(true)
            && predicate.map(|p| p == self.predicate).unwrap_or(true)
            && object.map(|o| o == self.object).unwrap_or(true)
    }
}

// ---------------------------------------------------------------------------
// GraphView
// ---------------------------------------------------------------------------

/// A read-only, named view over a set of RDF triples.
///
/// Conceptually represents a named graph in an RDF dataset.  The view holds an
/// owned snapshot of the triples (or references to an external store via index).
#[derive(Debug, Clone)]
pub struct GraphView {
    /// Name / IRI of this graph view.
    pub name: String,
    triples: Vec<RdfTriple>,
    /// Optional base IRI for relative IRI resolution.
    pub base_iri: Option<String>,
}

impl GraphView {
    /// Create a new named graph view with the given triples.
    pub fn new(name: &str, triples: Vec<RdfTriple>) -> Self {
        Self {
            name: name.to_string(),
            triples,
            base_iri: None,
        }
    }

    /// Set the base IRI for this view.
    pub fn with_base_iri(mut self, base: &str) -> Self {
        self.base_iri = Some(base.to_string());
        self
    }

    /// All triples in this view.
    pub fn triples(&self) -> &[RdfTriple] {
        &self.triples
    }

    /// Number of triples in this view.
    pub fn len(&self) -> usize {
        self.triples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }

    /// Find all triples matching the given triple pattern.  `None` = wildcard.
    pub fn find(
        &self,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
    ) -> Vec<&RdfTriple> {
        self.triples
            .iter()
            .filter(|t| t.matches(subject, predicate, object))
            .collect()
    }

    /// Returns `true` if the view contains `triple`.
    pub fn contains(&self, triple: &RdfTriple) -> bool {
        self.triples.contains(triple)
    }

    /// All distinct subjects in this view.
    pub fn subjects(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.triples.iter().map(|t| t.subject.as_str()).collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// All distinct predicates in this view.
    pub fn predicates(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.triples.iter().map(|t| t.predicate.as_str()).collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// All distinct objects in this view.
    pub fn objects(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.triples.iter().map(|t| t.object.as_str()).collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Create a [`FilteredView`] over this view.
    pub fn filter(
        &self,
        subject: Option<String>,
        predicate: Option<String>,
        object: Option<String>,
    ) -> FilteredView {
        FilteredView::new(self.clone(), subject, predicate, object)
    }
}

// ---------------------------------------------------------------------------
// FilteredView
// ---------------------------------------------------------------------------

/// A view that applies subject/predicate/object filters to a base [`GraphView`].
///
/// Filters are applied lazily — the underlying triples are not copied until
/// [`FilteredView::materialize`] is called.
#[derive(Debug, Clone)]
pub struct FilteredView {
    base: GraphView,
    subject_filter: Option<String>,
    predicate_filter: Option<String>,
    object_filter: Option<String>,
    /// Cached materialized result; `None` if not yet materialized.
    materialized: Option<Vec<RdfTriple>>,
}

impl FilteredView {
    /// Create a new filtered view.  Pass `None` for a wildcard slot.
    pub fn new(
        base: GraphView,
        subject: Option<String>,
        predicate: Option<String>,
        object: Option<String>,
    ) -> Self {
        Self {
            base,
            subject_filter: subject,
            predicate_filter: predicate,
            object_filter: object,
            materialized: None,
        }
    }

    /// Evaluate the filter and return matching triples (without caching).
    pub fn evaluate(&self) -> Vec<&RdfTriple> {
        self.base.find(
            self.subject_filter.as_deref(),
            self.predicate_filter.as_deref(),
            self.object_filter.as_deref(),
        )
    }

    /// Materialize the filtered result into an owned `Vec<RdfTriple>` and cache it.
    pub fn materialize(&mut self) -> &[RdfTriple] {
        if self.materialized.is_none() {
            self.materialized = Some(self.evaluate().into_iter().cloned().collect());
        }
        self.materialized.as_deref().unwrap_or(&[])
    }

    /// Invalidate the cached materialization (e.g. when the base view changes).
    pub fn invalidate(&mut self) {
        self.materialized = None;
    }

    /// Whether the cache is valid.
    pub fn is_cached(&self) -> bool {
        self.materialized.is_some()
    }

    /// Number of matching triples (evaluates without caching).
    pub fn count(&self) -> usize {
        self.evaluate().len()
    }

    /// Name of the underlying graph view.
    pub fn graph_name(&self) -> &str {
        &self.base.name
    }

    /// Add an additional predicate constraint (AND semantics).
    pub fn and_predicate(mut self, predicate: &str) -> Self {
        self.predicate_filter = Some(predicate.to_string());
        self.materialized = None;
        self
    }

    /// Add an additional object constraint (AND semantics).
    pub fn and_object(mut self, object: &str) -> Self {
        self.object_filter = Some(object.to_string());
        self.materialized = None;
        self
    }
}

// ---------------------------------------------------------------------------
// UnionView
// ---------------------------------------------------------------------------

/// A virtual view that presents the union of multiple named graph views.
///
/// Duplicate triples (appearing in more than one source graph) are deduplicated
/// when `deduplicate` is set to `true`.
#[derive(Debug, Clone)]
pub struct UnionView {
    /// Human-readable name for this union view.
    pub name: String,
    graphs: Vec<GraphView>,
    deduplicate: bool,
}

impl UnionView {
    /// Create a new union view over the given graphs.
    pub fn new(name: &str, graphs: Vec<GraphView>, deduplicate: bool) -> Self {
        Self {
            name: name.to_string(),
            graphs,
            deduplicate,
        }
    }

    /// Add a new graph to the union.
    pub fn add_graph(&mut self, graph: GraphView) {
        self.graphs.push(graph);
    }

    /// Enumerate all triples from all source graphs.
    ///
    /// If `deduplicate` is set, duplicate triples across graphs are returned only once.
    pub fn triples(&self) -> Vec<&RdfTriple> {
        let mut result: Vec<&RdfTriple> = Vec::new();
        for g in &self.graphs {
            for t in g.triples() {
                if self.deduplicate {
                    if !result.contains(&t) {
                        result.push(t);
                    }
                } else {
                    result.push(t);
                }
            }
        }
        result
    }

    /// Find triples matching a pattern across all source graphs.
    pub fn find(
        &self,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
    ) -> Vec<&RdfTriple> {
        let mut result: Vec<&RdfTriple> = Vec::new();
        for g in &self.graphs {
            for t in g.find(subject, predicate, object) {
                if self.deduplicate {
                    if !result.contains(&t) {
                        result.push(t);
                    }
                } else {
                    result.push(t);
                }
            }
        }
        result
    }

    /// Total triple count (may count duplicates if `deduplicate` is false).
    pub fn len(&self) -> usize {
        self.triples().len()
    }

    pub fn is_empty(&self) -> bool {
        self.graphs.iter().all(|g| g.is_empty())
    }

    /// Number of source graphs.
    pub fn graph_count(&self) -> usize {
        self.graphs.len()
    }

    /// Names of all source graphs.
    pub fn graph_names(&self) -> Vec<&str> {
        self.graphs.iter().map(|g| g.name.as_str()).collect()
    }
}

// ---------------------------------------------------------------------------
// MergedView
// ---------------------------------------------------------------------------

/// Merges a default graph with one or more named graphs.
///
/// The result presents all triples from the default graph plus those from
/// the named graphs.  The default graph takes priority: if a triple appears
/// in both the default graph and a named graph, only the default version is
/// returned when `deduplicate` is true.
#[derive(Debug, Clone)]
pub struct MergedView {
    /// Name for this merged view.
    pub name: String,
    default_graph: GraphView,
    named_graphs: Vec<GraphView>,
    deduplicate: bool,
    /// Materialized (cached) triple set.
    materialized: Option<Vec<RdfTriple>>,
}

impl MergedView {
    /// Create a new merged view from a default graph and a set of named graphs.
    pub fn new(
        name: &str,
        default_graph: GraphView,
        named_graphs: Vec<GraphView>,
        deduplicate: bool,
    ) -> Self {
        Self {
            name: name.to_string(),
            default_graph,
            named_graphs,
            deduplicate,
            materialized: None,
        }
    }

    /// Add a named graph to the merge.
    pub fn add_named_graph(&mut self, graph: GraphView) {
        self.named_graphs.push(graph);
        self.materialized = None;
    }

    /// Enumerate all triples.
    pub fn triples(&self) -> Vec<RdfTriple> {
        let mut result: Vec<RdfTriple> = self.default_graph.triples().to_vec();
        for ng in &self.named_graphs {
            for t in ng.triples() {
                if self.deduplicate {
                    if !result.contains(t) {
                        result.push(t.clone());
                    }
                } else {
                    result.push(t.clone());
                }
            }
        }
        result
    }

    /// Materialize the merged result into a cached `Vec<RdfTriple>`.
    pub fn materialize(&mut self) -> &[RdfTriple] {
        if self.materialized.is_none() {
            self.materialized = Some(self.triples());
        }
        self.materialized.as_deref().unwrap_or(&[])
    }

    /// Invalidate the materialization cache.
    pub fn invalidate(&mut self) {
        self.materialized = None;
    }

    /// Find triples matching a pattern across default + named graphs.
    pub fn find(
        &self,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
    ) -> Vec<RdfTriple> {
        self.triples()
            .into_iter()
            .filter(|t| t.matches(subject, predicate, object))
            .collect()
    }

    /// Total triple count in the merged view.
    pub fn len(&self) -> usize {
        self.triples().len()
    }

    pub fn is_empty(&self) -> bool {
        self.default_graph.is_empty() && self.named_graphs.iter().all(|g| g.is_empty())
    }

    /// Summary of how many triples come from each source.
    pub fn source_summary(&self) -> HashMap<String, usize> {
        let mut map = HashMap::new();
        map.insert(self.default_graph.name.clone(), self.default_graph.len());
        for ng in &self.named_graphs {
            map.insert(ng.name.clone(), ng.len());
        }
        map
    }
}

// ---------------------------------------------------------------------------
// ViewMaterializer
// ---------------------------------------------------------------------------

/// Caches and manages materialized views by name.
///
/// Each view is associated with an optional predicate dependency list.
/// When a triple with a tracked predicate changes, the affected views
/// are marked stale and must be re-materialized.
#[derive(Debug, Default)]
pub struct ViewMaterializer {
    graph_views: HashMap<String, GraphView>,
    filtered_cache: HashMap<String, Vec<RdfTriple>>,
    predicate_deps: HashMap<String, Vec<String>>, // view_name → [predicates]
    stale: std::collections::HashSet<String>,
}

impl ViewMaterializer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a `GraphView` under its name.
    pub fn register_graph_view(&mut self, view: GraphView, predicates: Vec<String>) {
        let name = view.name.clone();
        self.predicate_deps.insert(name.clone(), predicates);
        self.graph_views.insert(name.clone(), view);
        self.stale.remove(&name);
    }

    /// Retrieve a registered graph view by name.
    pub fn get_graph_view(&self, name: &str) -> Option<&GraphView> {
        self.graph_views.get(name)
    }

    /// Mark all views that depend on any of the given predicates as stale.
    pub fn mark_stale_for_predicates(&mut self, changed_predicates: &[String]) -> Vec<String> {
        let mut stale_views = Vec::new();
        for (view_name, deps) in &self.predicate_deps {
            let affected = deps.is_empty() || deps.iter().any(|d| changed_predicates.contains(d));
            if affected {
                self.stale.insert(view_name.clone());
                stale_views.push(view_name.clone());
            }
        }
        stale_views
    }

    /// Re-materialize a stale view with fresh triple data.
    pub fn refresh_view(&mut self, name: &str, fresh_triples: Vec<RdfTriple>) {
        if let Some(view) = self.graph_views.get_mut(name) {
            *view = GraphView::new(name, fresh_triples.clone());
        }
        self.filtered_cache.insert(name.to_string(), fresh_triples);
        self.stale.remove(name);
    }

    /// Whether `name` is currently marked stale.
    pub fn is_stale(&self, name: &str) -> bool {
        self.stale.contains(name)
    }

    /// All stale view names.
    pub fn stale_views(&self) -> Vec<&str> {
        self.stale.iter().map(|s| s.as_str()).collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_triple(s: &str, p: &str, o: &str) -> RdfTriple {
        RdfTriple::new(s, p, o)
    }

    fn make_view(name: &str, triples: &[(&str, &str, &str)]) -> GraphView {
        GraphView::new(
            name,
            triples
                .iter()
                .map(|(s, p, o)| make_triple(s, p, o))
                .collect(),
        )
    }

    // ---- RdfTriple ----

    #[test]
    fn test_rdf_triple_matches_wildcard() {
        let t = make_triple("s", "p", "o");
        assert!(t.matches(None, None, None));
    }

    #[test]
    fn test_rdf_triple_matches_bound_subject() {
        let t = make_triple("s", "p", "o");
        assert!(t.matches(Some("s"), None, None));
        assert!(!t.matches(Some("x"), None, None));
    }

    #[test]
    fn test_rdf_triple_matches_all_bound() {
        let t = make_triple("s", "p", "o");
        assert!(t.matches(Some("s"), Some("p"), Some("o")));
        assert!(!t.matches(Some("s"), Some("p"), Some("WRONG")));
    }

    // ---- GraphView ----

    #[test]
    fn test_graph_view_empty() {
        let v = GraphView::new("g", vec![]);
        assert!(v.is_empty());
        assert_eq!(v.len(), 0);
    }

    #[test]
    fn test_graph_view_triples() {
        let v = make_view("g", &[("s", "p", "o")]);
        assert_eq!(v.triples().len(), 1);
    }

    #[test]
    fn test_graph_view_find_by_predicate() {
        let v = make_view("g", &[("alice", "knows", "bob"), ("alice", "age", "30")]);
        let found = v.find(None, Some("knows"), None);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].object, "bob");
    }

    #[test]
    fn test_graph_view_find_no_match() {
        let v = make_view("g", &[("s", "p", "o")]);
        let found = v.find(None, Some("noSuchPredicate"), None);
        assert!(found.is_empty());
    }

    #[test]
    fn test_graph_view_contains() {
        let v = make_view("g", &[("s", "p", "o")]);
        assert!(v.contains(&make_triple("s", "p", "o")));
        assert!(!v.contains(&make_triple("s", "p", "X")));
    }

    #[test]
    fn test_graph_view_subjects() {
        let v = make_view(
            "g",
            &[("alice", "p", "o"), ("bob", "p", "o"), ("alice", "q", "x")],
        );
        let mut subjects = v.subjects();
        subjects.sort();
        assert_eq!(subjects, vec!["alice", "bob"]);
    }

    #[test]
    fn test_graph_view_predicates() {
        let v = make_view("g", &[("s", "p1", "o"), ("s", "p2", "o"), ("s", "p1", "x")]);
        let mut preds = v.predicates();
        preds.sort();
        assert_eq!(preds, vec!["p1", "p2"]);
    }

    #[test]
    fn test_graph_view_objects() {
        let v = make_view("g", &[("s", "p", "o1"), ("s", "p", "o2")]);
        let mut objs = v.objects();
        objs.sort();
        assert_eq!(objs, vec!["o1", "o2"]);
    }

    #[test]
    fn test_graph_view_with_base_iri() {
        let v = GraphView::new("g", vec![]).with_base_iri("http://example.org/");
        assert_eq!(v.base_iri.as_deref(), Some("http://example.org/"));
    }

    #[test]
    fn test_graph_view_filter_returns_filtered_view() {
        let v = make_view("g", &[("s", "p", "o"), ("s", "q", "x")]);
        let mut fv = v.filter(None, Some("p".to_string()), None);
        let mats = fv.materialize();
        assert_eq!(mats.len(), 1);
        assert_eq!(mats[0].predicate, "p");
    }

    // ---- FilteredView ----

    #[test]
    fn test_filtered_view_evaluate_empty() {
        let v = make_view("g", &[]);
        let fv = FilteredView::new(v, None, None, None);
        assert_eq!(fv.evaluate().len(), 0);
    }

    #[test]
    fn test_filtered_view_evaluate_subject_filter() {
        let v = make_view("g", &[("alice", "p", "o"), ("bob", "p", "o")]);
        let fv = FilteredView::new(v, Some("alice".to_string()), None, None);
        assert_eq!(fv.count(), 1);
    }

    #[test]
    fn test_filtered_view_materialize() {
        let v = make_view("g", &[("s", "p1", "o"), ("s", "p2", "o")]);
        let mut fv = FilteredView::new(v, None, Some("p1".to_string()), None);
        let result = fv.materialize();
        assert_eq!(result.len(), 1);
        assert!(fv.is_cached());
    }

    #[test]
    fn test_filtered_view_invalidate() {
        let v = make_view("g", &[("s", "p", "o")]);
        let mut fv = FilteredView::new(v, None, None, None);
        fv.materialize();
        assert!(fv.is_cached());
        fv.invalidate();
        assert!(!fv.is_cached());
    }

    #[test]
    fn test_filtered_view_graph_name() {
        let v = make_view("my_graph", &[]);
        let fv = FilteredView::new(v, None, None, None);
        assert_eq!(fv.graph_name(), "my_graph");
    }

    #[test]
    fn test_filtered_view_and_predicate() {
        let v = make_view("g", &[("s", "p1", "o"), ("s", "p2", "o")]);
        let fv = FilteredView::new(v, None, None, None).and_predicate("p1");
        assert_eq!(fv.count(), 1);
    }

    #[test]
    fn test_filtered_view_and_object() {
        let v = make_view("g", &[("s", "p", "o1"), ("s", "p", "o2")]);
        let fv = FilteredView::new(v, None, None, None).and_object("o1");
        assert_eq!(fv.count(), 1);
    }

    // ---- UnionView ----

    #[test]
    fn test_union_view_empty_graphs() {
        let uv = UnionView::new("u", vec![], false);
        assert!(uv.is_empty());
        assert_eq!(uv.len(), 0);
    }

    #[test]
    fn test_union_view_single_graph() {
        let g = make_view("g1", &[("s", "p", "o")]);
        let uv = UnionView::new("u", vec![g], false);
        assert_eq!(uv.len(), 1);
    }

    #[test]
    fn test_union_view_two_graphs_no_dedup() {
        let g1 = make_view("g1", &[("s", "p", "o")]);
        let g2 = make_view("g2", &[("s", "p", "o")]);
        let uv = UnionView::new("u", vec![g1, g2], false);
        assert_eq!(uv.len(), 2);
    }

    #[test]
    fn test_union_view_two_graphs_with_dedup() {
        let g1 = make_view("g1", &[("s", "p", "o")]);
        let g2 = make_view("g2", &[("s", "p", "o"), ("s2", "p2", "o2")]);
        let uv = UnionView::new("u", vec![g1, g2], true);
        assert_eq!(uv.len(), 2); // ("s","p","o") deduplicated; ("s2","p2","o2") unique
    }

    #[test]
    fn test_union_view_find() {
        let g1 = make_view("g1", &[("alice", "knows", "bob")]);
        let g2 = make_view("g2", &[("carol", "knows", "dave")]);
        let uv = UnionView::new("u", vec![g1, g2], false);
        let found = uv.find(None, Some("knows"), None);
        assert_eq!(found.len(), 2);
    }

    #[test]
    fn test_union_view_add_graph() {
        let mut uv = UnionView::new("u", vec![], false);
        assert_eq!(uv.graph_count(), 0);
        uv.add_graph(make_view("g1", &[("s", "p", "o")]));
        assert_eq!(uv.graph_count(), 1);
    }

    #[test]
    fn test_union_view_graph_names() {
        let g1 = make_view("graph_a", &[]);
        let g2 = make_view("graph_b", &[]);
        let uv = UnionView::new("u", vec![g1, g2], false);
        let mut names = uv.graph_names();
        names.sort();
        assert_eq!(names, vec!["graph_a", "graph_b"]);
    }

    // ---- MergedView ----

    #[test]
    fn test_merged_view_empty() {
        let default_g = GraphView::new("default", vec![]);
        let mv = MergedView::new("m", default_g, vec![], false);
        assert!(mv.is_empty());
        assert_eq!(mv.len(), 0);
    }

    #[test]
    fn test_merged_view_default_only() {
        let default_g = make_view("default", &[("s", "p", "o")]);
        let mv = MergedView::new("m", default_g, vec![], false);
        assert_eq!(mv.len(), 1);
    }

    #[test]
    fn test_merged_view_named_graphs() {
        let default_g = make_view("default", &[("s1", "p", "o1")]);
        let ng = make_view("named", &[("s2", "p", "o2")]);
        let mv = MergedView::new("m", default_g, vec![ng], false);
        assert_eq!(mv.len(), 2);
    }

    #[test]
    fn test_merged_view_dedup() {
        let default_g = make_view("default", &[("s", "p", "o")]);
        let ng = make_view("named", &[("s", "p", "o"), ("s2", "p2", "o2")]);
        let mv = MergedView::new("m", default_g, vec![ng], true);
        assert_eq!(mv.len(), 2); // ("s","p","o") deduped; ("s2","p2","o2") unique
    }

    #[test]
    fn test_merged_view_find() {
        let default_g = make_view("default", &[("alice", "type", "Person")]);
        let ng = make_view("named", &[("bob", "type", "Animal")]);
        let mv = MergedView::new("m", default_g, vec![ng], false);
        let found = mv.find(None, Some("type"), None);
        assert_eq!(found.len(), 2);
    }

    #[test]
    fn test_merged_view_materialize() {
        let default_g = make_view("default", &[("s", "p", "o")]);
        let mut mv = MergedView::new("m", default_g, vec![], false);
        let mats = mv.materialize();
        assert_eq!(mats.len(), 1);
    }

    #[test]
    fn test_merged_view_invalidate() {
        let default_g = make_view("default", &[("s", "p", "o")]);
        let mut mv = MergedView::new("m", default_g, vec![], false);
        mv.materialize();
        mv.invalidate();
        // Re-materialize should still work
        let mats = mv.materialize();
        assert_eq!(mats.len(), 1);
    }

    #[test]
    fn test_merged_view_source_summary() {
        let default_g = make_view("default", &[("s", "p", "o")]);
        let ng = make_view("ng1", &[("s2", "p2", "o2"), ("s3", "p3", "o3")]);
        let mv = MergedView::new("m", default_g, vec![ng], false);
        let summary = mv.source_summary();
        assert_eq!(*summary.get("default").unwrap(), 1);
        assert_eq!(*summary.get("ng1").unwrap(), 2);
    }

    #[test]
    fn test_merged_view_add_named_graph() {
        let default_g = make_view("default", &[("s", "p", "o")]);
        let mut mv = MergedView::new("m", default_g, vec![], false);
        assert_eq!(mv.len(), 1);
        mv.add_named_graph(make_view("extra", &[("s2", "p2", "o2")]));
        assert_eq!(mv.len(), 2);
    }

    // ---- ViewMaterializer ----

    #[test]
    fn test_view_materializer_empty() {
        let vm = ViewMaterializer::new();
        assert!(vm.stale_views().is_empty());
    }

    #[test]
    fn test_view_materializer_register_and_get() {
        let mut vm = ViewMaterializer::new();
        let view = make_view("v1", &[("s", "p", "o")]);
        vm.register_graph_view(view, vec!["p".to_string()]);
        assert!(vm.get_graph_view("v1").is_some());
    }

    #[test]
    fn test_view_materializer_mark_stale() {
        let mut vm = ViewMaterializer::new();
        let view = make_view("v1", &[]);
        vm.register_graph_view(view, vec!["http://p/age".to_string()]);
        let stale = vm.mark_stale_for_predicates(&["http://p/age".to_string()]);
        assert_eq!(stale.len(), 1);
        assert!(vm.is_stale("v1"));
    }

    #[test]
    fn test_view_materializer_no_stale_on_unrelated_predicate() {
        let mut vm = ViewMaterializer::new();
        let view = make_view("v1", &[]);
        vm.register_graph_view(view, vec!["http://p/name".to_string()]);
        let stale = vm.mark_stale_for_predicates(&["http://p/age".to_string()]);
        assert!(stale.is_empty());
        assert!(!vm.is_stale("v1"));
    }

    #[test]
    fn test_view_materializer_refresh_clears_stale() {
        let mut vm = ViewMaterializer::new();
        let view = make_view("v1", &[]);
        vm.register_graph_view(view, vec!["p".to_string()]);
        vm.mark_stale_for_predicates(&["p".to_string()]);
        assert!(vm.is_stale("v1"));
        vm.refresh_view("v1", vec![make_triple("s", "p", "o")]);
        assert!(!vm.is_stale("v1"));
    }

    #[test]
    fn test_view_materializer_empty_deps_always_stale() {
        let mut vm = ViewMaterializer::new();
        let view = make_view("v_all", &[]);
        vm.register_graph_view(view, vec![]); // no deps → any change stales it
        let stale = vm.mark_stale_for_predicates(&["any_predicate".to_string()]);
        assert!(stale.contains(&"v_all".to_string()));
    }
}
