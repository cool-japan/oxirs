//! SPARQL Update operation executor.
//!
//! Implements the SPARQL 1.1 Update operations (INSERT DATA, DELETE DATA,
//! DELETE WHERE, CLEAR, DROP, CREATE, INSERT/DELETE) over an in-memory quad store.
//! Supports atomic execution sequences with rollback on error.

use std::collections::{HashMap, HashSet};

/// A concrete quad (subject, predicate, object, optional graph).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Quad {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    /// `None` means the default graph.
    pub graph: Option<String>,
}

impl Quad {
    /// Create a quad in the default graph.
    pub fn default_graph(s: &str, p: &str, o: &str) -> Self {
        Quad {
            subject: s.to_string(),
            predicate: p.to_string(),
            object: o.to_string(),
            graph: None,
        }
    }

    /// Create a quad in a named graph.
    pub fn named_graph(s: &str, p: &str, o: &str, g: &str) -> Self {
        Quad {
            subject: s.to_string(),
            predicate: p.to_string(),
            object: o.to_string(),
            graph: Some(g.to_string()),
        }
    }
}

/// A term in a triple pattern (may be a concrete term or a variable).
#[derive(Debug, Clone, PartialEq)]
pub enum PatternTerm {
    Iri(String),
    Literal(String),
    Variable(String),
    Blank(String),
}

/// A triple pattern used in WHERE / DELETE WHERE clauses.
#[derive(Debug, Clone)]
pub struct TriplePattern {
    pub subject: PatternTerm,
    pub predicate: PatternTerm,
    pub object: PatternTerm,
}

/// A quad template used in INSERT clauses (graph is optional pattern term).
#[derive(Debug, Clone)]
pub struct QuadTemplate {
    pub subject: PatternTerm,
    pub predicate: PatternTerm,
    pub object: PatternTerm,
    pub graph: Option<PatternTerm>,
}

/// Target graph specifier for CLEAR / DROP.
#[derive(Debug, Clone)]
pub enum GraphTarget {
    Named(String),
    Default,
    All,
    NamedGraphs,
}

/// A SPARQL Update operation.
#[derive(Debug, Clone)]
pub enum UpdateOperation {
    /// INSERT DATA { quads }
    InsertData { quads: Vec<Quad> },
    /// DELETE DATA { quads }
    DeleteData { quads: Vec<Quad> },
    /// DELETE WHERE { patterns } (in optional named graph)
    DeleteWhere {
        graph: Option<String>,
        patterns: Vec<TriplePattern>,
    },
    /// CLEAR GRAPH / DEFAULT / ALL / NAMED
    Clear { graph: GraphTarget },
    /// DROP GRAPH (silent ignores GraphNotFound)
    Drop { graph: GraphTarget, silent: bool },
    /// CREATE GRAPH
    Create { graph: String, silent: bool },
    /// DELETE { patterns } INSERT { templates } WHERE { patterns }
    InsertDelete {
        delete_patterns: Vec<TriplePattern>,
        insert_templates: Vec<QuadTemplate>,
        where_patterns: Vec<TriplePattern>,
        using_graph: Option<String>,
    },
}

/// Summary of what a single (or sequence of) operation(s) accomplished.
#[derive(Debug, Default)]
pub struct UpdateResult {
    pub operations_applied: usize,
    pub triples_inserted: usize,
    pub triples_deleted: usize,
}

/// Errors that can occur during SPARQL Update execution.
#[derive(Debug)]
pub enum UpdateError {
    InvalidIri(String),
    GraphNotFound(String),
    GraphAlreadyExists(String),
    VariableInDeleteData(String),
}

impl std::fmt::Display for UpdateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UpdateError::InvalidIri(s) => write!(f, "Invalid IRI: {}", s),
            UpdateError::GraphNotFound(s) => write!(f, "Graph not found: {}", s),
            UpdateError::GraphAlreadyExists(s) => write!(f, "Graph already exists: {}", s),
            UpdateError::VariableInDeleteData(s) => {
                write!(f, "Variable not allowed in DELETE DATA: {}", s)
            }
        }
    }
}

impl std::error::Error for UpdateError {}

/// In-memory quad store that executes SPARQL Update operations.
///
/// The store maps `Option<String>` (graph name) to sets of (s, p, o) triples.
/// `None` represents the default graph. Named graphs must be explicitly created
/// (or are auto-created on first INSERT DATA).
#[derive(Debug, Default)]
pub struct UpdateProcessor {
    store: HashMap<Option<String>, HashSet<(String, String, String)>>,
}

impl UpdateProcessor {
    /// Create an empty update processor with an empty default graph.
    pub fn new() -> Self {
        let mut store = HashMap::new();
        store.insert(None, HashSet::new()); // ensure default graph exists
        UpdateProcessor { store }
    }

    /// Execute a single SPARQL Update operation.
    pub fn execute(&mut self, op: UpdateOperation) -> Result<UpdateResult, UpdateError> {
        let mut result = UpdateResult::default();
        self.apply_op(op, &mut result)?;
        Ok(result)
    }

    /// Execute a sequence of operations atomically.
    ///
    /// If any operation fails, all changes made so far in this call are rolled back.
    pub fn execute_sequence(
        &mut self,
        ops: Vec<UpdateOperation>,
    ) -> Result<UpdateResult, UpdateError> {
        let backup = self.store.clone();
        let mut result = UpdateResult::default();

        for op in ops {
            if let Err(e) = self.apply_op(op, &mut result) {
                // Rollback
                self.store = backup;
                return Err(e);
            }
        }
        Ok(result)
    }

    /// Total number of triples across all graphs (including default).
    pub fn triple_count(&self) -> usize {
        self.store.values().map(|s| s.len()).sum()
    }

    /// Number of named graphs (not counting the default graph).
    pub fn graph_count(&self) -> usize {
        self.store.keys().filter(|k| k.is_some()).count()
    }

    /// List all named graphs (not the default graph).
    pub fn list_graphs(&self) -> Vec<String> {
        let mut graphs: Vec<String> = self.store.keys().filter_map(|k| k.clone()).collect();
        graphs.sort();
        graphs
    }

    /// Check if a triple is present in the given graph.
    pub fn contains(&self, graph: Option<&str>, s: &str, p: &str, o: &str) -> bool {
        let key = graph.map(|g| g.to_string());
        self.store
            .get(&key)
            .is_some_and(|triples| triples.contains(&(s.to_string(), p.to_string(), o.to_string())))
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    fn apply_op(
        &mut self,
        op: UpdateOperation,
        result: &mut UpdateResult,
    ) -> Result<(), UpdateError> {
        result.operations_applied += 1;
        match op {
            UpdateOperation::InsertData { quads } => self.op_insert_data(quads, result),
            UpdateOperation::DeleteData { quads } => self.op_delete_data(quads, result),
            UpdateOperation::DeleteWhere { graph, patterns } => {
                self.op_delete_where(graph.as_deref(), &patterns, result)
            }
            UpdateOperation::Clear { graph } => self.op_clear(graph, result),
            UpdateOperation::Drop { graph, silent } => self.op_drop(graph, silent, result),
            UpdateOperation::Create { graph, silent } => self.op_create(&graph, silent, result),
            UpdateOperation::InsertDelete {
                delete_patterns,
                insert_templates,
                where_patterns,
                using_graph,
            } => self.op_insert_delete(
                &delete_patterns,
                &insert_templates,
                &where_patterns,
                using_graph.as_deref(),
                result,
            ),
        }
    }

    fn op_insert_data(
        &mut self,
        quads: Vec<Quad>,
        result: &mut UpdateResult,
    ) -> Result<(), UpdateError> {
        for quad in quads {
            let graph_entry = self
                .store
                .entry(quad.graph.clone())
                .or_insert_with(HashSet::new);
            let triple = (quad.subject, quad.predicate, quad.object);
            if graph_entry.insert(triple) {
                result.triples_inserted += 1;
            }
        }
        Ok(())
    }

    fn op_delete_data(
        &mut self,
        quads: Vec<Quad>,
        result: &mut UpdateResult,
    ) -> Result<(), UpdateError> {
        // DELETE DATA must not contain variables
        for quad in &quads {
            for term in [&quad.subject, &quad.predicate, &quad.object] {
                if term.starts_with('?') {
                    return Err(UpdateError::VariableInDeleteData(term.clone()));
                }
            }
        }
        for quad in quads {
            let key = quad.graph.clone();
            if let Some(graph_entry) = self.store.get_mut(&key) {
                let triple = (quad.subject, quad.predicate, quad.object);
                if graph_entry.remove(&triple) {
                    result.triples_deleted += 1;
                }
            }
        }
        Ok(())
    }

    fn op_delete_where(
        &mut self,
        graph: Option<&str>,
        patterns: &[TriplePattern],
        result: &mut UpdateResult,
    ) -> Result<(), UpdateError> {
        let key = graph.map(|g| g.to_string());
        let matching: Vec<(String, String, String)> = if let Some(triples) = self.store.get(&key) {
            triples
                .iter()
                .filter(|t| patterns.iter().all(|p| Self::match_pattern(t, p)))
                .cloned()
                .collect()
        } else {
            vec![]
        };
        if let Some(graph_entry) = self.store.get_mut(&key) {
            for triple in matching {
                if graph_entry.remove(&triple) {
                    result.triples_deleted += 1;
                }
            }
        }
        Ok(())
    }

    fn op_clear(
        &mut self,
        target: GraphTarget,
        result: &mut UpdateResult,
    ) -> Result<(), UpdateError> {
        match target {
            GraphTarget::Named(g) => {
                let key = Some(g.clone());
                if let Some(triples) = self.store.get_mut(&key) {
                    result.triples_deleted += triples.len();
                    triples.clear();
                }
            }
            GraphTarget::Default => {
                if let Some(triples) = self.store.get_mut(&None) {
                    result.triples_deleted += triples.len();
                    triples.clear();
                }
            }
            GraphTarget::All => {
                for triples in self.store.values_mut() {
                    result.triples_deleted += triples.len();
                    triples.clear();
                }
            }
            GraphTarget::NamedGraphs => {
                for (key, triples) in self.store.iter_mut() {
                    if key.is_some() {
                        result.triples_deleted += triples.len();
                        triples.clear();
                    }
                }
            }
        }
        Ok(())
    }

    fn op_drop(
        &mut self,
        target: GraphTarget,
        silent: bool,
        result: &mut UpdateResult,
    ) -> Result<(), UpdateError> {
        match target {
            GraphTarget::Named(g) => {
                let key = Some(g.clone());
                if self.store.remove(&key).is_some() {
                    result.triples_deleted += 0; // already counted by removing
                } else if !silent {
                    return Err(UpdateError::GraphNotFound(g));
                }
            }
            GraphTarget::Default => {
                if let Some(triples) = self.store.get_mut(&None) {
                    result.triples_deleted += triples.len();
                    triples.clear();
                }
            }
            GraphTarget::All => {
                let named_keys: Vec<Option<String>> =
                    self.store.keys().filter(|k| k.is_some()).cloned().collect();
                for key in named_keys {
                    self.store.remove(&key);
                }
                if let Some(triples) = self.store.get_mut(&None) {
                    result.triples_deleted += triples.len();
                    triples.clear();
                }
            }
            GraphTarget::NamedGraphs => {
                let named_keys: Vec<Option<String>> =
                    self.store.keys().filter(|k| k.is_some()).cloned().collect();
                for key in named_keys {
                    self.store.remove(&key);
                }
            }
        }
        Ok(())
    }

    fn op_create(
        &mut self,
        graph: &str,
        silent: bool,
        _result: &mut UpdateResult,
    ) -> Result<(), UpdateError> {
        let key = Some(graph.to_string());
        if let std::collections::hash_map::Entry::Vacant(e) = self.store.entry(key) {
            e.insert(HashSet::new());
        } else if !silent {
            return Err(UpdateError::GraphAlreadyExists(graph.to_string()));
        }
        Ok(())
    }

    fn op_insert_delete(
        &mut self,
        delete_patterns: &[TriplePattern],
        insert_templates: &[QuadTemplate],
        where_patterns: &[TriplePattern],
        using_graph: Option<&str>,
        result: &mut UpdateResult,
    ) -> Result<(), UpdateError> {
        // Evaluate WHERE clause to get bindings
        let key = using_graph.map(|g| g.to_string());
        let matched: Vec<(String, String, String)> = if let Some(triples) = self.store.get(&key) {
            triples
                .iter()
                .filter(|t| where_patterns.iter().all(|p| Self::match_pattern(t, p)))
                .cloned()
                .collect()
        } else {
            vec![]
        };

        // Build bindings from matched triples (simplified: bind ?s ?p ?o from WHERE patterns)
        let bindings: Vec<HashMap<String, String>> = matched
            .iter()
            .map(|(s, p, o)| {
                let mut m = HashMap::new();
                for pat in where_patterns {
                    if let PatternTerm::Variable(v) = &pat.subject {
                        m.insert(v.clone(), s.clone());
                    }
                    if let PatternTerm::Variable(v) = &pat.predicate {
                        m.insert(v.clone(), p.clone());
                    }
                    if let PatternTerm::Variable(v) = &pat.object {
                        m.insert(v.clone(), o.clone());
                    }
                }
                m
            })
            .collect();

        // Apply DELETE templates using bindings
        let mut to_delete: Vec<(Option<String>, String, String, String)> = vec![];
        for binding in &bindings {
            for pat in delete_patterns {
                if let (Some(s), Some(p), Some(o)) = (
                    Self::resolve_term(&pat.subject, binding),
                    Self::resolve_term(&pat.predicate, binding),
                    Self::resolve_term(&pat.object, binding),
                ) {
                    to_delete.push((key.clone(), s, p, o));
                }
            }
        }
        for (g, s, p, o) in to_delete {
            if let Some(triples) = self.store.get_mut(&g) {
                if triples.remove(&(s, p, o)) {
                    result.triples_deleted += 1;
                }
            }
        }

        // Apply INSERT templates using bindings
        for binding in &bindings {
            for tmpl in insert_templates {
                let graph_key = tmpl
                    .graph
                    .as_ref()
                    .and_then(|g| Self::resolve_term(g, binding));

                if let (Some(s), Some(p), Some(o)) = (
                    Self::resolve_term(&tmpl.subject, binding),
                    Self::resolve_term(&tmpl.predicate, binding),
                    Self::resolve_term(&tmpl.object, binding),
                ) {
                    let triples = self.store.entry(graph_key).or_insert_with(HashSet::new);
                    if triples.insert((s, p, o)) {
                        result.triples_inserted += 1;
                    }
                }
            }
        }

        Ok(())
    }

    /// Returns true if the triple matches ALL provided patterns (AND semantics).
    fn match_pattern(triple: &(String, String, String), pattern: &TriplePattern) -> bool {
        let (s, p, o) = triple;
        let match_term = |term: &PatternTerm, value: &str| -> bool {
            match term {
                PatternTerm::Variable(_) => true,
                PatternTerm::Iri(v) | PatternTerm::Literal(v) | PatternTerm::Blank(v) => v == value,
            }
        };
        match_term(&pattern.subject, s)
            && match_term(&pattern.predicate, p)
            && match_term(&pattern.object, o)
    }

    /// Resolve a PatternTerm to a concrete string using variable bindings.
    fn resolve_term(term: &PatternTerm, bindings: &HashMap<String, String>) -> Option<String> {
        match term {
            PatternTerm::Variable(v) => bindings.get(v).cloned(),
            PatternTerm::Iri(s) | PatternTerm::Literal(s) | PatternTerm::Blank(s) => {
                Some(s.clone())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iri(s: &str) -> String {
        s.to_string()
    }

    // ── InsertData ────────────────────────────────────────────────────────────

    #[test]
    fn test_insert_data_default_graph() {
        let mut proc = UpdateProcessor::new();
        proc.execute(UpdateOperation::InsertData {
            quads: vec![Quad::default_graph("s1", "p1", "o1")],
        })
        .unwrap();
        assert!(proc.contains(None, "s1", "p1", "o1"));
    }

    #[test]
    fn test_insert_data_named_graph() {
        let mut proc = UpdateProcessor::new();
        proc.execute(UpdateOperation::InsertData {
            quads: vec![Quad::named_graph("s", "p", "o", "http://g1")],
        })
        .unwrap();
        assert!(proc.contains(Some("http://g1"), "s", "p", "o"));
    }

    #[test]
    fn test_insert_data_multiple_quads() {
        let mut proc = UpdateProcessor::new();
        let result = proc
            .execute(UpdateOperation::InsertData {
                quads: vec![
                    Quad::default_graph("s1", "p1", "o1"),
                    Quad::default_graph("s2", "p2", "o2"),
                    Quad::default_graph("s3", "p3", "o3"),
                ],
            })
            .unwrap();
        assert_eq!(result.triples_inserted, 3);
        assert_eq!(proc.triple_count(), 3);
    }

    #[test]
    fn test_insert_data_deduplication() {
        let mut proc = UpdateProcessor::new();
        proc.execute(UpdateOperation::InsertData {
            quads: vec![Quad::default_graph("s", "p", "o")],
        })
        .unwrap();
        let result = proc
            .execute(UpdateOperation::InsertData {
                quads: vec![Quad::default_graph("s", "p", "o")],
            })
            .unwrap();
        // Duplicate triple — not inserted again
        assert_eq!(result.triples_inserted, 0);
        assert_eq!(proc.triple_count(), 1);
    }

    #[test]
    fn test_insert_data_result_counts() {
        let mut proc = UpdateProcessor::new();
        let result = proc
            .execute(UpdateOperation::InsertData {
                quads: vec![Quad::default_graph("s", "p", "o")],
            })
            .unwrap();
        assert_eq!(result.operations_applied, 1);
        assert_eq!(result.triples_inserted, 1);
        assert_eq!(result.triples_deleted, 0);
    }

    // ── DeleteData ────────────────────────────────────────────────────────────

    #[test]
    fn test_delete_data_existing_triple() {
        let mut proc = UpdateProcessor::new();
        proc.execute(UpdateOperation::InsertData {
            quads: vec![Quad::default_graph("s", "p", "o")],
        })
        .unwrap();
        let result = proc
            .execute(UpdateOperation::DeleteData {
                quads: vec![Quad::default_graph("s", "p", "o")],
            })
            .unwrap();
        assert_eq!(result.triples_deleted, 1);
        assert!(!proc.contains(None, "s", "p", "o"));
    }

    #[test]
    fn test_delete_data_nonexistent_triple() {
        let mut proc = UpdateProcessor::new();
        let result = proc
            .execute(UpdateOperation::DeleteData {
                quads: vec![Quad::default_graph("s", "p", "o")],
            })
            .unwrap();
        assert_eq!(result.triples_deleted, 0);
    }

    #[test]
    fn test_delete_data_variable_error() {
        let mut proc = UpdateProcessor::new();
        let result = proc.execute(UpdateOperation::DeleteData {
            quads: vec![Quad::default_graph("?s", "p", "o")],
        });
        assert!(result.is_err());
        match result.unwrap_err() {
            UpdateError::VariableInDeleteData(v) => assert_eq!(v, "?s"),
            e => panic!("wrong error: {:?}", e),
        }
    }

    // ── DeleteWhere ───────────────────────────────────────────────────────────

    #[test]
    fn test_delete_where_all_variables() {
        let mut proc = UpdateProcessor::new();
        proc.execute(UpdateOperation::InsertData {
            quads: vec![
                Quad::default_graph("s1", "p1", "o1"),
                Quad::default_graph("s2", "p2", "o2"),
            ],
        })
        .unwrap();
        let result = proc
            .execute(UpdateOperation::DeleteWhere {
                graph: None,
                patterns: vec![TriplePattern {
                    subject: PatternTerm::Variable("s".into()),
                    predicate: PatternTerm::Variable("p".into()),
                    object: PatternTerm::Variable("o".into()),
                }],
            })
            .unwrap();
        assert_eq!(result.triples_deleted, 2);
        assert_eq!(proc.triple_count(), 0);
    }

    #[test]
    fn test_delete_where_specific_predicate() {
        let mut proc = UpdateProcessor::new();
        proc.execute(UpdateOperation::InsertData {
            quads: vec![
                Quad::default_graph("s1", "type", "A"),
                Quad::default_graph("s2", "type", "B"),
                Quad::default_graph("s3", "name", "foo"),
            ],
        })
        .unwrap();
        proc.execute(UpdateOperation::DeleteWhere {
            graph: None,
            patterns: vec![TriplePattern {
                subject: PatternTerm::Variable("s".into()),
                predicate: PatternTerm::Iri("type".into()),
                object: PatternTerm::Variable("o".into()),
            }],
        })
        .unwrap();
        assert_eq!(proc.triple_count(), 1);
        assert!(proc.contains(None, "s3", "name", "foo"));
    }

    #[test]
    fn test_delete_where_named_graph() {
        let mut proc = UpdateProcessor::new();
        proc.execute(UpdateOperation::InsertData {
            quads: vec![
                Quad::named_graph("s", "p", "o", "http://g"),
                Quad::default_graph("s", "p", "o"),
            ],
        })
        .unwrap();
        proc.execute(UpdateOperation::DeleteWhere {
            graph: Some("http://g".into()),
            patterns: vec![TriplePattern {
                subject: PatternTerm::Variable("s".into()),
                predicate: PatternTerm::Variable("p".into()),
                object: PatternTerm::Variable("o".into()),
            }],
        })
        .unwrap();
        assert!(!proc.contains(Some("http://g"), "s", "p", "o"));
        assert!(proc.contains(None, "s", "p", "o")); // default graph unaffected
    }

    // ── Clear ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_clear_default_graph() {
        let mut proc = UpdateProcessor::new();
        proc.execute(UpdateOperation::InsertData {
            quads: vec![
                Quad::default_graph("s1", "p", "o"),
                Quad::named_graph("s2", "p", "o", "http://g"),
            ],
        })
        .unwrap();
        proc.execute(UpdateOperation::Clear {
            graph: GraphTarget::Default,
        })
        .unwrap();
        assert_eq!(proc.triple_count(), 1); // named graph untouched
        assert!(proc.contains(Some("http://g"), "s2", "p", "o"));
    }

    #[test]
    fn test_clear_named_graph() {
        let mut proc = UpdateProcessor::new();
        proc.execute(UpdateOperation::InsertData {
            quads: vec![
                Quad::default_graph("s1", "p", "o"),
                Quad::named_graph("s2", "p", "o", "http://g"),
            ],
        })
        .unwrap();
        proc.execute(UpdateOperation::Clear {
            graph: GraphTarget::Named("http://g".into()),
        })
        .unwrap();
        assert_eq!(proc.triple_count(), 1); // default graph untouched
    }

    #[test]
    fn test_clear_all() {
        let mut proc = UpdateProcessor::new();
        proc.execute(UpdateOperation::InsertData {
            quads: vec![
                Quad::default_graph("s1", "p", "o"),
                Quad::named_graph("s2", "p", "o", "http://g1"),
                Quad::named_graph("s3", "p", "o", "http://g2"),
            ],
        })
        .unwrap();
        proc.execute(UpdateOperation::Clear {
            graph: GraphTarget::All,
        })
        .unwrap();
        assert_eq!(proc.triple_count(), 0);
    }

    #[test]
    fn test_clear_named_graphs_only() {
        let mut proc = UpdateProcessor::new();
        proc.execute(UpdateOperation::InsertData {
            quads: vec![
                Quad::default_graph("s1", "p", "o"),
                Quad::named_graph("s2", "p", "o", "http://g"),
            ],
        })
        .unwrap();
        proc.execute(UpdateOperation::Clear {
            graph: GraphTarget::NamedGraphs,
        })
        .unwrap();
        assert_eq!(proc.triple_count(), 1); // default graph unchanged
    }

    // ── Drop ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_drop_named_graph() {
        let mut proc = UpdateProcessor::new();
        proc.execute(UpdateOperation::Create {
            graph: "http://g".into(),
            silent: false,
        })
        .unwrap();
        assert_eq!(proc.graph_count(), 1);
        proc.execute(UpdateOperation::Drop {
            graph: GraphTarget::Named("http://g".into()),
            silent: false,
        })
        .unwrap();
        assert_eq!(proc.graph_count(), 0);
    }

    #[test]
    fn test_drop_nonexistent_silent_ok() {
        let mut proc = UpdateProcessor::new();
        let result = proc.execute(UpdateOperation::Drop {
            graph: GraphTarget::Named("http://missing".into()),
            silent: true,
        });
        assert!(result.is_ok());
    }

    #[test]
    fn test_drop_nonexistent_not_silent_error() {
        let mut proc = UpdateProcessor::new();
        let result = proc.execute(UpdateOperation::Drop {
            graph: GraphTarget::Named("http://missing".into()),
            silent: false,
        });
        assert!(result.is_err());
        match result.unwrap_err() {
            UpdateError::GraphNotFound(g) => assert_eq!(g, "http://missing"),
            e => panic!("wrong error: {:?}", e),
        }
    }

    #[test]
    fn test_drop_all() {
        let mut proc = UpdateProcessor::new();
        proc.execute(UpdateOperation::InsertData {
            quads: vec![
                Quad::default_graph("s", "p", "o"),
                Quad::named_graph("s", "p", "o", "http://g"),
            ],
        })
        .unwrap();
        proc.execute(UpdateOperation::Drop {
            graph: GraphTarget::All,
            silent: false,
        })
        .unwrap();
        assert_eq!(proc.triple_count(), 0);
        assert_eq!(proc.graph_count(), 0);
    }

    #[test]
    fn test_drop_named_graphs_only() {
        let mut proc = UpdateProcessor::new();
        proc.execute(UpdateOperation::InsertData {
            quads: vec![
                Quad::default_graph("s", "p", "o"),
                Quad::named_graph("s", "p", "o", "http://g"),
            ],
        })
        .unwrap();
        proc.execute(UpdateOperation::Drop {
            graph: GraphTarget::NamedGraphs,
            silent: false,
        })
        .unwrap();
        assert_eq!(proc.triple_count(), 1);
        assert_eq!(proc.graph_count(), 0);
    }

    // ── Create ────────────────────────────────────────────────────────────────

    #[test]
    fn test_create_graph() {
        let mut proc = UpdateProcessor::new();
        proc.execute(UpdateOperation::Create {
            graph: "http://new".into(),
            silent: false,
        })
        .unwrap();
        assert_eq!(proc.graph_count(), 1);
        assert!(proc.list_graphs().contains(&"http://new".to_string()));
    }

    #[test]
    fn test_create_duplicate_error() {
        let mut proc = UpdateProcessor::new();
        proc.execute(UpdateOperation::Create {
            graph: "http://g".into(),
            silent: false,
        })
        .unwrap();
        let result = proc.execute(UpdateOperation::Create {
            graph: "http://g".into(),
            silent: false,
        });
        assert!(result.is_err());
        match result.unwrap_err() {
            UpdateError::GraphAlreadyExists(g) => assert_eq!(g, "http://g"),
            e => panic!("wrong error: {:?}", e),
        }
    }

    #[test]
    fn test_create_duplicate_silent_ok() {
        let mut proc = UpdateProcessor::new();
        proc.execute(UpdateOperation::Create {
            graph: "http://g".into(),
            silent: false,
        })
        .unwrap();
        let result = proc.execute(UpdateOperation::Create {
            graph: "http://g".into(),
            silent: true,
        });
        assert!(result.is_ok());
    }

    // ── InsertDelete ──────────────────────────────────────────────────────────

    #[test]
    fn test_insert_delete_basic() {
        let mut proc = UpdateProcessor::new();
        // Insert starting data
        proc.execute(UpdateOperation::InsertData {
            quads: vec![Quad::default_graph("s", "age", "30")],
        })
        .unwrap();

        // DELETE { ?s age ?old } INSERT { ?s age ?old } WHERE { ?s age ?old }
        // (trivial case: re-inserts same data, deletes same data)
        let result = proc
            .execute(UpdateOperation::InsertDelete {
                delete_patterns: vec![TriplePattern {
                    subject: PatternTerm::Variable("s".into()),
                    predicate: PatternTerm::Iri("age".into()),
                    object: PatternTerm::Variable("old".into()),
                }],
                insert_templates: vec![QuadTemplate {
                    subject: PatternTerm::Variable("s".into()),
                    predicate: PatternTerm::Iri("age".into()),
                    object: PatternTerm::Literal("31".into()),
                    graph: None,
                }],
                where_patterns: vec![TriplePattern {
                    subject: PatternTerm::Variable("s".into()),
                    predicate: PatternTerm::Iri("age".into()),
                    object: PatternTerm::Variable("old".into()),
                }],
                using_graph: None,
            })
            .unwrap();
        assert!(result.triples_deleted >= 1);
        assert!(proc.contains(None, "s", "age", "31"));
    }

    // ── execute_sequence rollback ─────────────────────────────────────────────

    #[test]
    fn test_execute_sequence_success() {
        let mut proc = UpdateProcessor::new();
        let result = proc
            .execute_sequence(vec![
                UpdateOperation::InsertData {
                    quads: vec![Quad::default_graph("s1", "p", "o1")],
                },
                UpdateOperation::InsertData {
                    quads: vec![Quad::default_graph("s2", "p", "o2")],
                },
            ])
            .unwrap();
        assert_eq!(result.triples_inserted, 2);
        assert_eq!(proc.triple_count(), 2);
    }

    #[test]
    fn test_execute_sequence_rollback_on_error() {
        let mut proc = UpdateProcessor::new();
        // Pre-populate
        proc.execute(UpdateOperation::InsertData {
            quads: vec![Quad::default_graph("pre", "p", "o")],
        })
        .unwrap();

        let before_count = proc.triple_count();

        // sequence: valid insert + invalid delete-data with variable → rolls back
        let result = proc.execute_sequence(vec![
            UpdateOperation::InsertData {
                quads: vec![Quad::default_graph("new", "p", "o")],
            },
            UpdateOperation::DeleteData {
                quads: vec![Quad::default_graph("?variable", "p", "o")], // ERROR
            },
        ]);

        assert!(result.is_err());
        // State rolled back to before_count
        assert_eq!(proc.triple_count(), before_count);
    }

    #[test]
    fn test_execute_sequence_create_and_insert() {
        let mut proc = UpdateProcessor::new();
        proc.execute_sequence(vec![
            UpdateOperation::Create {
                graph: "http://g".into(),
                silent: false,
            },
            UpdateOperation::InsertData {
                quads: vec![Quad::named_graph("s", "p", "o", "http://g")],
            },
        ])
        .unwrap();
        assert_eq!(proc.graph_count(), 1);
        assert!(proc.contains(Some("http://g"), "s", "p", "o"));
    }

    // ── Metadata queries ──────────────────────────────────────────────────────

    #[test]
    fn test_triple_count_empty() {
        let proc = UpdateProcessor::new();
        assert_eq!(proc.triple_count(), 0);
    }

    #[test]
    fn test_graph_count_empty() {
        let proc = UpdateProcessor::new();
        assert_eq!(proc.graph_count(), 0);
    }

    #[test]
    fn test_list_graphs_empty() {
        let proc = UpdateProcessor::new();
        assert!(proc.list_graphs().is_empty());
    }

    #[test]
    fn test_list_graphs_multiple() {
        let mut proc = UpdateProcessor::new();
        proc.execute(UpdateOperation::Create {
            graph: "http://g2".into(),
            silent: false,
        })
        .unwrap();
        proc.execute(UpdateOperation::Create {
            graph: "http://g1".into(),
            silent: false,
        })
        .unwrap();
        let graphs = proc.list_graphs();
        assert_eq!(
            graphs,
            vec!["http://g1".to_string(), "http://g2".to_string()]
        );
    }

    #[test]
    fn test_contains_true() {
        let mut proc = UpdateProcessor::new();
        proc.execute(UpdateOperation::InsertData {
            quads: vec![Quad::default_graph("s", "p", "o")],
        })
        .unwrap();
        assert!(proc.contains(None, "s", "p", "o"));
    }

    #[test]
    fn test_contains_false() {
        let proc = UpdateProcessor::new();
        assert!(!proc.contains(None, "s", "p", "o"));
    }

    #[test]
    fn test_triple_count_after_delete() {
        let mut proc = UpdateProcessor::new();
        proc.execute(UpdateOperation::InsertData {
            quads: vec![
                Quad::default_graph("a", "p", "o"),
                Quad::default_graph("b", "p", "o"),
            ],
        })
        .unwrap();
        proc.execute(UpdateOperation::DeleteData {
            quads: vec![Quad::default_graph("a", "p", "o")],
        })
        .unwrap();
        assert_eq!(proc.triple_count(), 1);
    }
}
