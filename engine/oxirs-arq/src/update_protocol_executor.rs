//! In-memory executor for SPARQL 1.1 Update operations.
//!
//! [`UpdateExecutor`] is a minimal store-independent dataset executor used by
//! the standalone update protocol facade.  It maintains a *default graph*
//! (a `Vec<Triple>`) and an arbitrary number of *named graphs* keyed by IRI
//! string.  Pattern-based operations perform a simple structural match —
//! variables behave as wildcards that can bind to any term, and consistent
//! bindings across multiple patterns are intersected by shared variable name.

use std::collections::HashMap;

use crate::update_protocol_types::{
    ArqError, ClearType, DropType, PatternTerm, SparqlUpdate, Triple, TriplePattern, UpdateResult,
};

// ---------------------------------------------------------------------------
// In-memory UpdateExecutor
// ---------------------------------------------------------------------------

/// A minimal in-memory dataset executor for SPARQL 1.1 Update.
///
/// It maintains a *default graph* (a `Vec<Triple>`) and an arbitrary number
/// of *named graphs* keyed by IRI string.  Pattern-based operations perform a
/// simple structural match (no unification — variables are left as wildcards
/// that match anything).
pub struct UpdateExecutor {
    /// Triples in the default graph.
    pub(crate) triples: Vec<Triple>,
    /// Named graphs keyed by IRI.
    pub(crate) named_graphs: HashMap<String, Vec<Triple>>,
}

impl UpdateExecutor {
    /// Create an empty executor.
    pub fn new() -> Self {
        Self {
            triples: Vec::new(),
            named_graphs: HashMap::new(),
        }
    }

    /// Execute a single update and return a summary.
    pub fn execute(&mut self, update: &SparqlUpdate) -> Result<UpdateResult, ArqError> {
        match update {
            SparqlUpdate::InsertData(triples) => {
                let count = triples.len();
                self.triples.extend(triples.iter().cloned());
                Ok(UpdateResult {
                    triples_inserted: count,
                    triples_deleted: 0,
                    graphs_affected: 0,
                })
            }
            SparqlUpdate::DeleteData(triples) => {
                let before = self.triples.len();
                for t in triples {
                    self.triples.retain(|existing| existing != t);
                }
                let deleted = before - self.triples.len();
                Ok(UpdateResult {
                    triples_inserted: 0,
                    triples_deleted: deleted,
                    graphs_affected: 0,
                })
            }
            SparqlUpdate::InsertWhere {
                template,
                where_clause,
            } => {
                let bindings = self.match_patterns(where_clause);
                let mut inserted = 0usize;
                for binding in &bindings {
                    if let Some(triple) = instantiate_template_triple(template.first(), binding) {
                        for t in &triple {
                            if !self.triples.contains(t) {
                                self.triples.push(t.clone());
                                inserted += 1;
                            }
                        }
                    }
                }
                Ok(UpdateResult {
                    triples_inserted: inserted,
                    triples_deleted: 0,
                    graphs_affected: 0,
                })
            }
            SparqlUpdate::DeleteWhere { where_clause, .. } => {
                let bindings = self.match_patterns(where_clause);
                let to_delete: Vec<Triple> = bindings
                    .into_iter()
                    .filter_map(|b| {
                        let s = b.get("s").cloned()?;
                        let p = b.get("p").cloned()?;
                        let o = b.get("o").cloned()?;
                        Some(Triple::new(s, p, o))
                    })
                    .collect();
                let before = self.triples.len();
                for t in &to_delete {
                    self.triples.retain(|e| e != t);
                }
                let deleted = before - self.triples.len();
                Ok(UpdateResult {
                    triples_inserted: 0,
                    triples_deleted: deleted,
                    graphs_affected: 0,
                })
            }
            SparqlUpdate::Modify {
                delete,
                insert,
                where_clause,
            } => {
                let bindings = self.match_patterns(where_clause);
                let mut inserted = 0usize;
                let mut deleted_count = 0usize;
                for binding in &bindings {
                    // Delete first.
                    for tp in delete {
                        if let Some(t) = instantiate_one(tp, binding) {
                            let before = self.triples.len();
                            self.triples.retain(|e| e != &t);
                            deleted_count += before - self.triples.len();
                        }
                    }
                    // Then insert.
                    for tp in insert {
                        if let Some(t) = instantiate_one(tp, binding) {
                            if !self.triples.contains(&t) {
                                self.triples.push(t);
                                inserted += 1;
                            }
                        }
                    }
                }
                Ok(UpdateResult {
                    triples_inserted: inserted,
                    triples_deleted: deleted_count,
                    graphs_affected: 0,
                })
            }
            SparqlUpdate::CreateGraph { iri, silent } => {
                if self.named_graphs.contains_key(iri) && !silent {
                    return Err(ArqError(format!("graph <{iri}> already exists")));
                }
                self.named_graphs.entry(iri.clone()).or_default();
                Ok(UpdateResult {
                    triples_inserted: 0,
                    triples_deleted: 0,
                    graphs_affected: 1,
                })
            }
            SparqlUpdate::DropGraph {
                iri,
                silent,
                drop_type,
            } => {
                let count = match drop_type {
                    DropType::Graph => {
                        let key = iri.as_deref().unwrap_or("");
                        if self.named_graphs.remove(key).is_none() && !silent {
                            return Err(ArqError(format!("graph <{key}> does not exist")));
                        }
                        1
                    }
                    DropType::Default => {
                        self.triples.clear();
                        1
                    }
                    DropType::Named => {
                        let count = self.named_graphs.len();
                        self.named_graphs.clear();
                        count
                    }
                    DropType::All => {
                        let ng = self.named_graphs.len();
                        self.named_graphs.clear();
                        self.triples.clear();
                        ng + 1
                    }
                };
                Ok(UpdateResult {
                    triples_inserted: 0,
                    triples_deleted: 0,
                    graphs_affected: count,
                })
            }
            SparqlUpdate::ClearGraph {
                iri,
                silent,
                clear_type,
            } => {
                let count = match clear_type {
                    ClearType::Graph => {
                        let key = iri.as_deref().unwrap_or("");
                        match self.named_graphs.get_mut(key) {
                            Some(g) => {
                                g.clear();
                                1
                            }
                            None if *silent => 0,
                            None => return Err(ArqError(format!("graph <{key}> does not exist"))),
                        }
                    }
                    ClearType::Default => {
                        self.triples.clear();
                        1
                    }
                    ClearType::Named => {
                        for g in self.named_graphs.values_mut() {
                            g.clear();
                        }
                        self.named_graphs.len()
                    }
                    ClearType::All => {
                        self.triples.clear();
                        for g in self.named_graphs.values_mut() {
                            g.clear();
                        }
                        self.named_graphs.len() + 1
                    }
                };
                Ok(UpdateResult {
                    triples_inserted: 0,
                    triples_deleted: 0,
                    graphs_affected: count,
                })
            }
            SparqlUpdate::CopyGraph {
                source,
                target,
                silent: _,
            } => {
                let src_triples: Vec<Triple> =
                    self.named_graphs.get(source).cloned().unwrap_or_default();
                let count = src_triples.len();
                let tgt = self.named_graphs.entry(target.clone()).or_default();
                tgt.clear();
                tgt.extend(src_triples);
                Ok(UpdateResult {
                    triples_inserted: count,
                    triples_deleted: 0,
                    graphs_affected: 1,
                })
            }
            SparqlUpdate::MoveGraph {
                source,
                target,
                silent: _,
            } => {
                let src_triples = self.named_graphs.remove(source).unwrap_or_default();
                let count = src_triples.len();
                let tgt = self.named_graphs.entry(target.clone()).or_default();
                tgt.clear();
                tgt.extend(src_triples);
                Ok(UpdateResult {
                    triples_inserted: count,
                    triples_deleted: 0,
                    graphs_affected: 2,
                })
            }
            SparqlUpdate::AddGraph {
                source,
                target,
                silent: _,
            } => {
                let src_triples: Vec<Triple> =
                    self.named_graphs.get(source).cloned().unwrap_or_default();
                let count = src_triples.len();
                let tgt = self.named_graphs.entry(target.clone()).or_default();
                tgt.extend(src_triples);
                Ok(UpdateResult {
                    triples_inserted: count,
                    triples_deleted: 0,
                    graphs_affected: 1,
                })
            }
            SparqlUpdate::Load { iri, into, silent } => {
                // Actual HTTP loading is not implemented in this in-memory executor.
                // Return success (silent) or error (non-silent).
                if *silent {
                    Ok(UpdateResult::default())
                } else {
                    Err(ArqError(format!(
                        "LOAD is not supported in the in-memory executor (iri=<{iri}>, into={into:?})"
                    )))
                }
            }
        }
    }

    /// Execute a sequence of update operations and collect their results.
    pub fn execute_all(&mut self, updates: &[SparqlUpdate]) -> Result<Vec<UpdateResult>, ArqError> {
        updates.iter().map(|u| self.execute(u)).collect()
    }

    /// Number of triples in the default graph.
    pub fn triple_count(&self) -> usize {
        self.triples.len()
    }

    /// Number of named graphs (not counting the default graph).
    pub fn graph_count(&self) -> usize {
        self.named_graphs.len()
    }

    /// Return the triples in a named graph, or `None` if it does not exist.
    pub fn get_graph(&self, iri: &str) -> Option<&Vec<Triple>> {
        self.named_graphs.get(iri)
    }

    /// Return a reference to the default graph's triple set.
    pub fn default_graph(&self) -> &Vec<Triple> {
        &self.triples
    }
}

impl Default for UpdateExecutor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Pattern matching helpers
// ---------------------------------------------------------------------------

pub(crate) type Binding = HashMap<String, String>;

/// Match a slice of triple patterns against the default graph, returning all
/// consistent variable bindings.  Each pattern is matched independently and
/// bindings from consecutive patterns are intersected by joining on shared
/// variable names.
fn match_patterns(triples: &[Triple], patterns: &[TriplePattern]) -> Vec<Binding> {
    let mut results: Vec<Binding> = vec![HashMap::new()];

    for pattern in patterns {
        let mut next: Vec<Binding> = Vec::new();
        for binding in &results {
            for triple in triples {
                if let Some(new_binding) = match_pattern(triple, pattern, binding) {
                    next.push(new_binding);
                }
            }
        }
        results = next;
    }

    results
}

/// Try to extend `existing_binding` with the variable bindings produced by
/// matching `triple` against `pattern`.  Returns `None` on conflict.
fn match_pattern(triple: &Triple, pattern: &TriplePattern, existing: &Binding) -> Option<Binding> {
    let mut binding = existing.clone();
    bind_term(&triple.s, &pattern.s, &mut binding)?;
    bind_term(&triple.p, &pattern.p, &mut binding)?;
    bind_term(&triple.o, &pattern.o, &mut binding)?;
    Some(binding)
}

/// Attempt to bind `value` against `term`, extending `binding` if `term` is a
/// variable.  Returns `None` when an existing binding is inconsistent.
fn bind_term(value: &str, term: &PatternTerm, binding: &mut Binding) -> Option<()> {
    match term {
        PatternTerm::Variable(var) => {
            if let Some(existing) = binding.get(var.as_str()) {
                if existing != value {
                    return None;
                }
            } else {
                binding.insert(var.clone(), value.to_string());
            }
            Some(())
        }
        PatternTerm::Iri(iri) => {
            if iri == value {
                Some(())
            } else {
                None
            }
        }
        PatternTerm::Literal(lit) => {
            // Compare the content without surrounding quotes.
            let inner = lit.trim_matches('"').trim_matches('\'');
            if inner == value || lit == value {
                Some(())
            } else {
                None
            }
        }
        PatternTerm::BlankNode(bn) => {
            if bn == value {
                Some(())
            } else {
                None
            }
        }
    }
}

impl UpdateExecutor {
    /// Match patterns against the default graph's triple set.
    fn match_patterns(&self, patterns: &[TriplePattern]) -> Vec<Binding> {
        match_patterns(&self.triples, patterns)
    }
}

/// Try to instantiate a single `TriplePattern` against a `Binding`, producing
/// a `Triple` when all positions resolve to concrete terms.
fn instantiate_one(pattern: &TriplePattern, binding: &Binding) -> Option<Triple> {
    let s = resolve_term(&pattern.s, binding)?;
    let p = resolve_term(&pattern.p, binding)?;
    let o = resolve_term(&pattern.o, binding)?;
    Some(Triple::new(s, p, o))
}

/// Try to instantiate the first `TriplePattern` in `templates`, returning a
/// `Vec<Triple>` (0 or 1 elements).  This helper is used for `InsertWhere`.
fn instantiate_template_triple(
    template: Option<&TriplePattern>,
    binding: &Binding,
) -> Option<Vec<Triple>> {
    let tp = template?;
    Some(instantiate_one(tp, binding).into_iter().collect())
}

/// Resolve a `PatternTerm` to a concrete string using `binding`.  Returns
/// `None` when a variable is unbound.
fn resolve_term(term: &PatternTerm, binding: &Binding) -> Option<String> {
    match term {
        PatternTerm::Variable(var) => binding.get(var.as_str()).cloned(),
        PatternTerm::Iri(iri) => Some(iri.clone()),
        PatternTerm::Literal(lit) => Some(lit.trim_matches('"').trim_matches('\'').to_string()),
        PatternTerm::BlankNode(bn) => Some(bn.clone()),
    }
}
