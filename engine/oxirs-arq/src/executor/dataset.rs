//! Dataset Implementation
//!
//! This module provides dataset abstractions and implementations for query execution.

use crate::algebra::{PropertyPath, Term as AlgebraTerm, TriplePattern};
use crate::path::{PathDataset, PropertyPath as PathPropertyPath};
use anyhow::{anyhow, Result};
use oxirs_core::model::{GraphName, NamedNode};
use oxirs_core::RdfTerm;
use std::collections::HashSet;

/// Selects which graph(s) a triple-pattern lookup targets within a
/// [`Dataset`].
///
/// This mirrors the SPARQL RDF-dataset distinction between the single active
/// *default graph* and an individually addressed *named graph*. It deliberately
/// does not offer an "all graphs / union" mode, because unioning every graph
/// for a plain pattern is exactly the semantic bug this abstraction exists to
/// prevent (see [`Dataset::find_triples`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphSelector {
    /// The dataset's active default graph.
    DefaultGraph,
    /// A single named graph addressed by its IRI (e.g. `GRAPH <iri> { ... }`).
    Named(NamedNode),
}

/// Dataset trait for data access during query execution
pub trait Dataset: Send + Sync {
    /// Find all triples matching the given pattern in the dataset's **active
    /// default graph**.
    ///
    /// Per SPARQL RDF-dataset semantics a plain (non-`GRAPH`) basic graph
    /// pattern reads only the default graph, not the union of every named
    /// graph. Implementors backed by a quad store MUST scope this to the
    /// default graph rather than passing an unbound graph filter.
    fn find_triples(
        &self,
        pattern: &TriplePattern,
    ) -> Result<Vec<(AlgebraTerm, AlgebraTerm, AlgebraTerm)>>;

    /// Check if a triple exists in the dataset
    fn contains_triple(
        &self,
        subject: &AlgebraTerm,
        predicate: &AlgebraTerm,
        object: &AlgebraTerm,
    ) -> Result<bool>;

    /// Get all subjects in the dataset
    fn subjects(&self) -> Result<Vec<AlgebraTerm>>;

    /// Get all predicates in the dataset
    fn predicates(&self) -> Result<Vec<AlgebraTerm>>;

    /// Get all objects in the dataset
    fn objects(&self) -> Result<Vec<AlgebraTerm>>;

    /// Whether this dataset can honor graph-scoped access
    /// ([`Dataset::find_triples_in`]) and named-graph enumeration
    /// ([`Dataset::named_graphs`]).
    ///
    /// Defaults to `false`; graph-capable implementors override it to `true`.
    fn has_graph_support(&self) -> bool {
        false
    }

    /// Find all triples matching `pattern` within the graph selected by
    /// `selector`.
    ///
    /// The default implementation FAILS LOUD: a dataset that cannot scope by
    /// graph must never silently fall back to unioning every graph, since that
    /// corrupts `GRAPH` / `FROM` semantics. Graph-capable implementors override
    /// this (and set [`Dataset::has_graph_support`] to `true`).
    fn find_triples_in(
        &self,
        selector: &GraphSelector,
        pattern: &TriplePattern,
    ) -> Result<Vec<(AlgebraTerm, AlgebraTerm, AlgebraTerm)>> {
        let _ = (selector, pattern);
        Err(anyhow!(
            "graph-scoped access not supported by this dataset: it does not \
             implement Dataset::find_triples_in (has_graph_support() is false)"
        ))
    }

    /// Enumerate the named graphs visible in this dataset, as `Term::Iri`
    /// values, for `GRAPH ?g` evaluation.
    ///
    /// The default implementation FAILS LOUD for the same reason as
    /// [`Dataset::find_triples_in`].
    fn named_graphs(&self) -> Result<Vec<AlgebraTerm>> {
        Err(anyhow!(
            "named-graph enumeration not supported by this dataset: it does \
             not implement Dataset::named_graphs (has_graph_support() is false)"
        ))
    }
}

/// In-memory dataset implementation for testing.
///
/// Each stored triple carries an optional graph label: `None` places it in the
/// default graph, `Some(iri)` in the named graph `iri`. This lets the in-memory
/// dataset exercise `GRAPH` / `FROM` scoping without a backing quad store.
#[derive(Debug, Clone, Default)]
pub struct InMemoryDataset {
    quads: Vec<(Option<NamedNode>, AlgebraTerm, AlgebraTerm, AlgebraTerm)>,
}

impl InMemoryDataset {
    pub fn new() -> Self {
        Self { quads: Vec::new() }
    }

    /// Add a triple to the **default graph**.
    pub fn add_triple(
        &mut self,
        subject: AlgebraTerm,
        predicate: AlgebraTerm,
        object: AlgebraTerm,
    ) {
        self.quads.push((None, subject, predicate, object));
    }

    /// Add a triple to the named graph `graph`.
    pub fn add_triple_in_graph(
        &mut self,
        graph: NamedNode,
        subject: AlgebraTerm,
        predicate: AlgebraTerm,
        object: AlgebraTerm,
    ) {
        self.quads.push((Some(graph), subject, predicate, object));
    }

    /// Build a default-graph-only dataset from a list of triples.
    pub fn from_triples(triples: Vec<(AlgebraTerm, AlgebraTerm, AlgebraTerm)>) -> Self {
        Self {
            quads: triples
                .into_iter()
                .map(|(s, p, o)| (None, s, p, o))
                .collect(),
        }
    }

    /// Filter stored quads by an explicit graph target and the pattern.
    fn scan(
        &self,
        want_graph: Option<&NamedNode>,
        pattern: &TriplePattern,
    ) -> Vec<(AlgebraTerm, AlgebraTerm, AlgebraTerm)> {
        self.quads
            .iter()
            .filter(|(g, s, p, o)| {
                let graph_ok = match (want_graph, g) {
                    (None, None) => true,
                    (Some(want), Some(have)) => want == have,
                    _ => false,
                };
                graph_ok
                    && matches_term(&pattern.subject, s)
                    && matches_term(&pattern.predicate, p)
                    && matches_term(&pattern.object, o)
            })
            .map(|(_, s, p, o)| (s.clone(), p.clone(), o.clone()))
            .collect()
    }
}

impl Dataset for InMemoryDataset {
    fn find_triples(
        &self,
        pattern: &TriplePattern,
    ) -> Result<Vec<(AlgebraTerm, AlgebraTerm, AlgebraTerm)>> {
        // Plain BGP reads the default graph only.
        Ok(self.scan(None, pattern))
    }

    fn find_triples_in(
        &self,
        selector: &GraphSelector,
        pattern: &TriplePattern,
    ) -> Result<Vec<(AlgebraTerm, AlgebraTerm, AlgebraTerm)>> {
        let want = match selector {
            GraphSelector::DefaultGraph => None,
            GraphSelector::Named(iri) => Some(iri),
        };
        Ok(self.scan(want, pattern))
    }

    fn has_graph_support(&self) -> bool {
        true
    }

    fn named_graphs(&self) -> Result<Vec<AlgebraTerm>> {
        let mut seen: HashSet<NamedNode> = HashSet::new();
        let mut names: Vec<AlgebraTerm> = Vec::new();
        for (g, _, _, _) in &self.quads {
            if let Some(iri) = g {
                if seen.insert(iri.clone()) {
                    names.push(AlgebraTerm::Iri(iri.clone()));
                }
            }
        }
        Ok(names)
    }

    fn contains_triple(
        &self,
        subject: &AlgebraTerm,
        predicate: &AlgebraTerm,
        object: &AlgebraTerm,
    ) -> Result<bool> {
        // Existence check against the default graph, consistent with
        // `find_triples`.
        Ok(self
            .quads
            .iter()
            .any(|(g, s, p, o)| g.is_none() && s == subject && p == predicate && o == object))
    }

    fn subjects(&self) -> Result<Vec<AlgebraTerm>> {
        let subjects: HashSet<_> = self.quads.iter().map(|(_, s, _, _)| s.clone()).collect();
        Ok(subjects.into_iter().collect())
    }

    fn predicates(&self) -> Result<Vec<AlgebraTerm>> {
        let predicates: HashSet<_> = self.quads.iter().map(|(_, _, p, _)| p.clone()).collect();
        Ok(predicates.into_iter().collect())
    }

    fn objects(&self) -> Result<Vec<AlgebraTerm>> {
        let objects: HashSet<_> = self.quads.iter().map(|(_, _, _, o)| o.clone()).collect();
        Ok(objects.into_iter().collect())
    }
}

fn matches_term(pattern: &AlgebraTerm, term: &AlgebraTerm) -> bool {
    match pattern {
        AlgebraTerm::Variable(_) => true, // Variables match any term
        _ => pattern == term,
    }
}

/// Adapter to make Dataset implement PathDataset
pub struct DatasetPathAdapter<'a> {
    dataset: &'a dyn Dataset,
}

impl<'a> DatasetPathAdapter<'a> {
    pub fn new(dataset: &'a dyn Dataset) -> Self {
        Self { dataset }
    }
}

impl<'a> PathDataset for DatasetPathAdapter<'a> {
    fn find_outgoing(
        &self,
        subject: &AlgebraTerm,
        predicate: &AlgebraTerm,
    ) -> Result<Vec<AlgebraTerm>> {
        let pattern = TriplePattern::new(
            subject.clone(),
            predicate.clone(),
            AlgebraTerm::Variable(crate::algebra::Variable::new("?o")?),
        );
        let triples = self.dataset.find_triples(&pattern)?;
        Ok(triples.into_iter().map(|(_, _, o)| o).collect())
    }

    fn find_incoming(
        &self,
        predicate: &AlgebraTerm,
        object: &AlgebraTerm,
    ) -> Result<Vec<AlgebraTerm>> {
        let pattern = TriplePattern::new(
            AlgebraTerm::Variable(crate::algebra::Variable::new("?s")?),
            predicate.clone(),
            object.clone(),
        );
        let triples = self.dataset.find_triples(&pattern)?;
        Ok(triples.into_iter().map(|(s, _, _)| s).collect())
    }

    fn find_predicates(
        &self,
        subject: &AlgebraTerm,
        object: &AlgebraTerm,
    ) -> Result<Vec<AlgebraTerm>> {
        let pattern = TriplePattern::new(
            subject.clone(),
            AlgebraTerm::Variable(crate::algebra::Variable::new("?p")?),
            object.clone(),
        );
        let triples = self.dataset.find_triples(&pattern)?;
        Ok(triples.into_iter().map(|(_, p, _)| p).collect())
    }

    fn get_predicates(&self) -> Result<Vec<AlgebraTerm>> {
        self.dataset.predicates()
    }

    fn contains_triple(
        &self,
        subject: &AlgebraTerm,
        predicate: &AlgebraTerm,
        object: &AlgebraTerm,
    ) -> Result<bool> {
        self.dataset.contains_triple(subject, predicate, object)
    }
}

/// Convert algebra PropertyPath to path module PropertyPath
pub fn convert_property_path(path: &PropertyPath) -> Result<PathPropertyPath> {
    match path {
        PropertyPath::Iri(iri) => Ok(PathPropertyPath::Direct(AlgebraTerm::Iri(iri.clone()))),
        PropertyPath::Variable(var) => {
            Ok(PathPropertyPath::Direct(AlgebraTerm::Variable(var.clone())))
        }
        PropertyPath::Inverse(inner) => {
            let inner_path = convert_property_path(inner)?;
            Ok(PathPropertyPath::Inverse(Box::new(inner_path)))
        }
        PropertyPath::Sequence(left, right) => {
            let left_path = convert_property_path(left)?;
            let right_path = convert_property_path(right)?;
            Ok(PathPropertyPath::Sequence(
                Box::new(left_path),
                Box::new(right_path),
            ))
        }
        PropertyPath::Alternative(left, right) => {
            let left_path = convert_property_path(left)?;
            let right_path = convert_property_path(right)?;
            Ok(PathPropertyPath::Alternative(
                Box::new(left_path),
                Box::new(right_path),
            ))
        }
        PropertyPath::ZeroOrMore(inner) => {
            let inner_path = convert_property_path(inner)?;
            Ok(PathPropertyPath::ZeroOrMore(Box::new(inner_path)))
        }
        PropertyPath::OneOrMore(inner) => {
            let inner_path = convert_property_path(inner)?;
            Ok(PathPropertyPath::OneOrMore(Box::new(inner_path)))
        }
        PropertyPath::ZeroOrOne(inner) => {
            let inner_path = convert_property_path(inner)?;
            Ok(PathPropertyPath::ZeroOrOne(Box::new(inner_path)))
        }
        PropertyPath::NegatedPropertySet(paths) => {
            let mut terms = Vec::new();
            for p in paths {
                match p {
                    PropertyPath::Iri(iri) => terms.push(AlgebraTerm::Iri(iri.clone())),
                    PropertyPath::Variable(var) => terms.push(AlgebraTerm::Variable(var.clone())),
                    _ => {
                        return Err(anyhow!(
                            "Negated property set can only contain IRIs or variables"
                        ))
                    }
                }
            }
            Ok(PathPropertyPath::NegatedPropertySet(terms))
        }
    }
}

/// Adapter to make ConcreteStore implement Dataset trait
/// This is primarily for benchmarking and testing purposes
pub struct ConcreteStoreDataset {
    store: std::sync::Arc<oxirs_core::rdf_store::ConcreteStore>,
}

impl ConcreteStoreDataset {
    pub fn new(store: oxirs_core::rdf_store::ConcreteStore) -> Self {
        Self {
            store: std::sync::Arc::new(store),
        }
    }

    pub fn from_arc(store: std::sync::Arc<oxirs_core::rdf_store::ConcreteStore>) -> Self {
        Self { store }
    }
}

impl Clone for ConcreteStoreDataset {
    fn clone(&self) -> Self {
        Self {
            store: std::sync::Arc::clone(&self.store),
        }
    }
}

impl Dataset for ConcreteStoreDataset {
    fn find_triples(
        &self,
        pattern: &TriplePattern,
    ) -> Result<Vec<(AlgebraTerm, AlgebraTerm, AlgebraTerm)>> {
        // Plain BGP reads the default graph only (SPARQL dataset semantics).
        self.find_triples_in(&GraphSelector::DefaultGraph, pattern)
    }

    fn find_triples_in(
        &self,
        selector: &GraphSelector,
        pattern: &TriplePattern,
    ) -> Result<Vec<(AlgebraTerm, AlgebraTerm, AlgebraTerm)>> {
        use oxirs_core::rdf_store::Store;

        let (subject, predicate, object) = pattern_to_query_terms(pattern)?;
        let graph_name = graph_selector_to_name(selector);
        let quads = self.store.find_quads(
            subject.as_ref(),
            predicate.as_ref(),
            object.as_ref(),
            Some(&graph_name),
        )?;
        Ok(quads_to_algebra_triples(quads))
    }

    fn has_graph_support(&self) -> bool {
        true
    }

    fn named_graphs(&self) -> Result<Vec<AlgebraTerm>> {
        collect_named_graphs(self.store.as_ref())
    }

    fn contains_triple(
        &self,
        subject: &AlgebraTerm,
        predicate: &AlgebraTerm,
        object: &AlgebraTerm,
    ) -> Result<bool> {
        let pattern = TriplePattern::new(subject.clone(), predicate.clone(), object.clone());
        let triples = self.find_triples(&pattern)?;
        Ok(!triples.is_empty())
    }

    fn subjects(&self) -> Result<Vec<AlgebraTerm>> {
        use oxirs_core::rdf_store::Store;
        let quads = self.store.find_quads(None, None, None, None)?;
        let subjects: HashSet<_> = quads
            .into_iter()
            .filter_map(|quad| match quad.subject() {
                oxirs_core::model::Subject::NamedNode(n) => Some(AlgebraTerm::Iri(
                    oxirs_core::model::NamedNode::new(n.as_str()).ok()?,
                )),
                oxirs_core::model::Subject::BlankNode(b) => {
                    Some(AlgebraTerm::BlankNode(b.as_str().to_string()))
                }
                oxirs_core::model::Subject::Variable(v) => Some(AlgebraTerm::Variable(v.clone())),
                oxirs_core::model::Subject::QuotedTriple(_) => None,
            })
            .collect();
        Ok(subjects.into_iter().collect())
    }

    fn predicates(&self) -> Result<Vec<AlgebraTerm>> {
        use oxirs_core::rdf_store::Store;
        let quads = self.store.find_quads(None, None, None, None)?;
        let predicates: HashSet<_> = quads
            .into_iter()
            .filter_map(|quad| {
                Some(AlgebraTerm::Iri(
                    oxirs_core::model::NamedNode::new(quad.predicate().as_str()).ok()?,
                ))
            })
            .collect();
        Ok(predicates.into_iter().collect())
    }

    fn objects(&self) -> Result<Vec<AlgebraTerm>> {
        use oxirs_core::rdf_store::Store;
        let quads = self.store.find_quads(None, None, None, None)?;
        let objects: HashSet<_> = quads
            .into_iter()
            .filter_map(|quad| match quad.object() {
                oxirs_core::model::Object::NamedNode(n) => Some(AlgebraTerm::Iri(
                    oxirs_core::model::NamedNode::new(n.as_str()).ok()?,
                )),
                oxirs_core::model::Object::Literal(l) => {
                    Some(AlgebraTerm::Literal(core_literal_to_algebra(l)))
                }
                oxirs_core::model::Object::BlankNode(b) => {
                    Some(AlgebraTerm::BlankNode(b.as_str().to_string()))
                }
                oxirs_core::model::Object::Variable(v) => Some(AlgebraTerm::Variable(v.clone())),
                oxirs_core::model::Object::QuotedTriple(_) => None,
            })
            .collect();
        Ok(objects.into_iter().collect())
    }
}

/// Well-known datatype IRIs whose presence is implicit for plain / language
/// literals and therefore represented as `datatype: None` in the algebra model.
const XSD_STRING: &str = "http://www.w3.org/2001/XMLSchema#string";
const RDF_LANG_STRING: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString";

/// Build a store `Literal` from an algebra `Literal`, preserving the datatype
/// and language tag. Mirrors `update::UpdateExecutor::term_to_object` so that
/// typed/lang-tagged literal patterns match real data in the store instead of
/// being silently coerced to plain `xsd:string`.
pub(crate) fn algebra_literal_to_core(lit: &crate::algebra::Literal) -> oxirs_core::model::Literal {
    if let Some(lang) = &lit.language {
        oxirs_core::model::Literal::new_language_tagged_literal(&lit.value, lang)
            .unwrap_or_else(|_| oxirs_core::model::Literal::new(&lit.value))
    } else if let Some(dt) = &lit.datatype {
        oxirs_core::model::Literal::new_typed(&lit.value, dt.clone())
    } else {
        oxirs_core::model::Literal::new(&lit.value)
    }
}

/// Convert a store `Literal` into an algebra `Literal`, preserving the datatype
/// and language tag (dropping only the implicit `xsd:string` / `rdf:langString`
/// datatypes, which the algebra model expresses via `datatype: None`).
pub(crate) fn core_literal_to_algebra(l: &oxirs_core::model::Literal) -> crate::algebra::Literal {
    if let Some(lang) = l.language() {
        crate::algebra::Literal {
            value: l.value().to_string(),
            language: Some(lang.to_string()),
            datatype: None,
        }
    } else {
        let dt = l.datatype().into_owned();
        let datatype = if dt.as_str() == XSD_STRING || dt.as_str() == RDF_LANG_STRING {
            None
        } else {
            Some(dt)
        };
        crate::algebra::Literal {
            value: l.value().to_string(),
            language: None,
            datatype,
        }
    }
}

/// Convert a triple pattern into optional (subject, predicate, object) store
/// query terms. `None` in any position means an unbound wildcard.
#[allow(clippy::type_complexity)]
pub(crate) fn pattern_to_query_terms(
    pattern: &TriplePattern,
) -> Result<(
    Option<oxirs_core::model::Subject>,
    Option<oxirs_core::model::Predicate>,
    Option<oxirs_core::model::Object>,
)> {
    let subject = match &pattern.subject {
        AlgebraTerm::Iri(iri) => Some(oxirs_core::model::Subject::NamedNode(iri.clone())),
        AlgebraTerm::Variable(_) => None,
        AlgebraTerm::BlankNode(id) => Some(oxirs_core::model::Subject::BlankNode(
            oxirs_core::model::BlankNode::new(id)
                .map_err(|e| anyhow!("Invalid blank node: {}", e))?,
        )),
        _ => return Err(anyhow!("Invalid subject in pattern")),
    };

    let predicate = match &pattern.predicate {
        AlgebraTerm::Iri(iri) => Some(oxirs_core::model::Predicate::NamedNode(iri.clone())),
        AlgebraTerm::Variable(_) => None,
        AlgebraTerm::PropertyPath(path) => match path {
            PropertyPath::Iri(iri) => Some(oxirs_core::model::Predicate::NamedNode(iri.clone())),
            PropertyPath::Variable(_) => None,
            _ => {
                return Err(anyhow!(
                    "Complex property paths not yet supported in find_triples"
                ))
            }
        },
        _ => {
            return Err(anyhow!(
                "Predicate must be IRI, variable, or property path, got: {:?}",
                pattern.predicate
            ))
        }
    };

    let object = match &pattern.object {
        AlgebraTerm::Iri(iri) => Some(oxirs_core::model::Object::NamedNode(iri.clone())),
        AlgebraTerm::Literal(lit) => Some(oxirs_core::model::Object::Literal(
            algebra_literal_to_core(lit),
        )),
        AlgebraTerm::BlankNode(id) => Some(oxirs_core::model::Object::BlankNode(
            oxirs_core::model::BlankNode::new(id)
                .map_err(|e| anyhow!("Invalid blank node: {}", e))?,
        )),
        AlgebraTerm::Variable(_) => None,
        _ => return Err(anyhow!("Invalid object in pattern")),
    };

    Ok((subject, predicate, object))
}

/// Convert store quads into algebra triples, preserving literal datatype and
/// language. RDF-star quoted triples are skipped.
pub(crate) fn quads_to_algebra_triples(
    quads: Vec<oxirs_core::model::Quad>,
) -> Vec<(AlgebraTerm, AlgebraTerm, AlgebraTerm)> {
    quads
        .into_iter()
        .filter_map(|quad| {
            let s = match quad.subject() {
                oxirs_core::model::Subject::NamedNode(n) => {
                    AlgebraTerm::Iri(oxirs_core::model::NamedNode::new(n.as_str()).ok()?)
                }
                oxirs_core::model::Subject::BlankNode(b) => {
                    AlgebraTerm::BlankNode(b.as_str().to_string())
                }
                oxirs_core::model::Subject::Variable(v) => AlgebraTerm::Variable(v.clone()),
                oxirs_core::model::Subject::QuotedTriple(_) => return None,
            };

            let p = AlgebraTerm::Iri(
                oxirs_core::model::NamedNode::new(quad.predicate().as_str()).ok()?,
            );

            let o = match quad.object() {
                oxirs_core::model::Object::NamedNode(n) => {
                    AlgebraTerm::Iri(oxirs_core::model::NamedNode::new(n.as_str()).ok()?)
                }
                oxirs_core::model::Object::Literal(l) => {
                    AlgebraTerm::Literal(core_literal_to_algebra(l))
                }
                oxirs_core::model::Object::BlankNode(b) => {
                    AlgebraTerm::BlankNode(b.as_str().to_string())
                }
                oxirs_core::model::Object::Variable(v) => AlgebraTerm::Variable(v.clone()),
                oxirs_core::model::Object::QuotedTriple(_) => return None,
            };

            Some((s, p, o))
        })
        .collect()
}

/// Map a [`GraphSelector`] to the concrete store [`GraphName`] it addresses.
pub(crate) fn graph_selector_to_name(selector: &GraphSelector) -> GraphName {
    match selector {
        GraphSelector::DefaultGraph => GraphName::DefaultGraph,
        GraphSelector::Named(iri) => GraphName::NamedNode(iri.clone()),
    }
}

/// Enumerate the distinct named graphs of a store as algebra `Term::Iri`
/// values.
///
/// Delegates to the store's [`Store::named_graphs`](oxirs_core::rdf_store::Store::named_graphs)
/// trait method, which both real oxirs-core stores (`RdfStore` and
/// `ConcreteStore`) override to read the interned graph-name index in
/// O(graphs). This replaces the earlier streaming `for_each_quad` scan
/// (O(quads)) that only existed because the trait method used to return an
/// empty vector; now that the concrete stores populate it for real, the trait
/// is the authoritative source.
pub(crate) fn collect_named_graphs(
    store: &dyn oxirs_core::rdf_store::Store,
) -> Result<Vec<AlgebraTerm>> {
    Ok(store
        .named_graphs()?
        .into_iter()
        .map(AlgebraTerm::Iri)
        .collect())
}

/// Adapter exposing any `&dyn oxirs_core::Store` as a query [`Dataset`].
///
/// This is the bridge used by `UPDATE ... WHERE` evaluation so that
/// DELETE/INSERT WHERE and DELETE-INSERT operate on the real triple store
/// (via `Store::find_quads`) rather than a disconnected in-memory executor.
pub struct StoreRefDataset<'s> {
    store: &'s dyn oxirs_core::rdf_store::Store,
}

impl<'s> StoreRefDataset<'s> {
    /// Wrap a borrowed store as a dataset.
    pub fn new(store: &'s dyn oxirs_core::rdf_store::Store) -> Self {
        Self { store }
    }
}

impl<'s> Dataset for StoreRefDataset<'s> {
    fn find_triples(
        &self,
        pattern: &TriplePattern,
    ) -> Result<Vec<(AlgebraTerm, AlgebraTerm, AlgebraTerm)>> {
        // Plain BGP reads the dataset's default graph only. Previously this
        // passed `None` (unbound graph), which UNIONED every named graph and
        // violated SPARQL default-graph semantics.
        self.find_triples_in(&GraphSelector::DefaultGraph, pattern)
    }

    fn find_triples_in(
        &self,
        selector: &GraphSelector,
        pattern: &TriplePattern,
    ) -> Result<Vec<(AlgebraTerm, AlgebraTerm, AlgebraTerm)>> {
        let (subject, predicate, object) = pattern_to_query_terms(pattern)?;
        let graph_name = graph_selector_to_name(selector);
        let quads = self.store.find_quads(
            subject.as_ref(),
            predicate.as_ref(),
            object.as_ref(),
            Some(&graph_name),
        )?;
        Ok(quads_to_algebra_triples(quads))
    }

    fn has_graph_support(&self) -> bool {
        true
    }

    fn named_graphs(&self) -> Result<Vec<AlgebraTerm>> {
        collect_named_graphs(self.store)
    }

    fn contains_triple(
        &self,
        subject: &AlgebraTerm,
        predicate: &AlgebraTerm,
        object: &AlgebraTerm,
    ) -> Result<bool> {
        let pattern = TriplePattern::new(subject.clone(), predicate.clone(), object.clone());
        Ok(!self.find_triples(&pattern)?.is_empty())
    }

    fn subjects(&self) -> Result<Vec<AlgebraTerm>> {
        let quads = self.store.find_quads(None, None, None, None)?;
        let subjects: HashSet<_> = quads
            .into_iter()
            .filter_map(|quad| match quad.subject() {
                oxirs_core::model::Subject::NamedNode(n) => Some(AlgebraTerm::Iri(
                    oxirs_core::model::NamedNode::new(n.as_str()).ok()?,
                )),
                oxirs_core::model::Subject::BlankNode(b) => {
                    Some(AlgebraTerm::BlankNode(b.as_str().to_string()))
                }
                oxirs_core::model::Subject::Variable(v) => Some(AlgebraTerm::Variable(v.clone())),
                oxirs_core::model::Subject::QuotedTriple(_) => None,
            })
            .collect();
        Ok(subjects.into_iter().collect())
    }

    fn predicates(&self) -> Result<Vec<AlgebraTerm>> {
        let quads = self.store.find_quads(None, None, None, None)?;
        let predicates: HashSet<_> = quads
            .into_iter()
            .filter_map(|quad| {
                Some(AlgebraTerm::Iri(
                    oxirs_core::model::NamedNode::new(quad.predicate().as_str()).ok()?,
                ))
            })
            .collect();
        Ok(predicates.into_iter().collect())
    }

    fn objects(&self) -> Result<Vec<AlgebraTerm>> {
        let quads = self.store.find_quads(None, None, None, None)?;
        let objects: HashSet<_> = quads
            .into_iter()
            .filter_map(|quad| match quad.object() {
                oxirs_core::model::Object::NamedNode(n) => Some(AlgebraTerm::Iri(
                    oxirs_core::model::NamedNode::new(n.as_str()).ok()?,
                )),
                oxirs_core::model::Object::Literal(l) => {
                    Some(AlgebraTerm::Literal(core_literal_to_algebra(l)))
                }
                oxirs_core::model::Object::BlankNode(b) => {
                    Some(AlgebraTerm::BlankNode(b.as_str().to_string()))
                }
                oxirs_core::model::Object::Variable(v) => Some(AlgebraTerm::Variable(v.clone())),
                oxirs_core::model::Object::QuotedTriple(_) => None,
            })
            .collect();
        Ok(objects.into_iter().collect())
    }
}

/// Build a wildcard triple pattern (`?s ?p ?o`) for whole-graph scans.
fn wildcard_pattern() -> TriplePattern {
    TriplePattern::new(
        AlgebraTerm::Variable(crate::algebra::Variable::new_unchecked("s")),
        AlgebraTerm::Variable(crate::algebra::Variable::new_unchecked("p")),
        AlgebraTerm::Variable(crate::algebra::Variable::new_unchecked("o")),
    )
}

/// A [`Dataset`] view that pins every plain lookup to a single graph.
///
/// Wrapping a base dataset in a `GraphScopedDataset` makes the *unscoped*
/// [`Dataset::find_triples`] entry point — the funnel that BGP evaluation,
/// property paths and `EXISTS` all pass through — read only the selected graph.
/// The inner pattern-evaluation code therefore needs no changes to honor
/// `GRAPH` scoping: it keeps calling `find_triples`, and the view redirects it.
///
/// A nested `GRAPH` inside a scoped view overrides the outer scope, exactly as
/// SPARQL requires: [`Dataset::find_triples_in`] forwards the inner selector to
/// the base rather than intersecting with the outer one.
pub struct GraphScopedDataset<'a> {
    base: &'a dyn Dataset,
    selector: GraphSelector,
}

impl<'a> GraphScopedDataset<'a> {
    /// Restrict all plain lookups on `base` to the graph named by `selector`.
    pub fn new(base: &'a dyn Dataset, selector: GraphSelector) -> Self {
        Self { base, selector }
    }
}

impl<'a> Dataset for GraphScopedDataset<'a> {
    fn find_triples(
        &self,
        pattern: &TriplePattern,
    ) -> Result<Vec<(AlgebraTerm, AlgebraTerm, AlgebraTerm)>> {
        self.base.find_triples_in(&self.selector, pattern)
    }

    fn find_triples_in(
        &self,
        selector: &GraphSelector,
        pattern: &TriplePattern,
    ) -> Result<Vec<(AlgebraTerm, AlgebraTerm, AlgebraTerm)>> {
        // Inner GRAPH wins over the outer scope.
        self.base.find_triples_in(selector, pattern)
    }

    fn has_graph_support(&self) -> bool {
        self.base.has_graph_support()
    }

    fn named_graphs(&self) -> Result<Vec<AlgebraTerm>> {
        self.base.named_graphs()
    }

    fn contains_triple(
        &self,
        subject: &AlgebraTerm,
        predicate: &AlgebraTerm,
        object: &AlgebraTerm,
    ) -> Result<bool> {
        let pattern = TriplePattern::new(subject.clone(), predicate.clone(), object.clone());
        Ok(!self.find_triples(&pattern)?.is_empty())
    }

    fn subjects(&self) -> Result<Vec<AlgebraTerm>> {
        let triples = self.find_triples(&wildcard_pattern())?;
        let set: HashSet<_> = triples.into_iter().map(|(s, _, _)| s).collect();
        Ok(set.into_iter().collect())
    }

    fn predicates(&self) -> Result<Vec<AlgebraTerm>> {
        let triples = self.find_triples(&wildcard_pattern())?;
        let set: HashSet<_> = triples.into_iter().map(|(_, p, _)| p).collect();
        Ok(set.into_iter().collect())
    }

    fn objects(&self) -> Result<Vec<AlgebraTerm>> {
        let triples = self.find_triples(&wildcard_pattern())?;
        let set: HashSet<_> = triples.into_iter().map(|(_, _, o)| o).collect();
        Ok(set.into_iter().collect())
    }
}

/// A [`Dataset`] view implementing SPARQL `FROM` / `FROM NAMED` dataset
/// construction over a base dataset.
///
/// * The active **default graph** is the union of the `FROM` graphs
///   (`default_graphs`); with no `FROM` graphs it falls back to the base
///   dataset's own default graph.
/// * The visible **named graphs** (for `GRAPH` and `GRAPH ?g` enumeration) are
///   exactly the `FROM NAMED` graphs (`named_graphs`); a `FROM`-only clause
///   therefore exposes no named graphs.
/// * An entirely empty clause is a transparent passthrough of the base's
///   semantics.
///
/// Construct with [`with_dataset_clause`] from a parsed
/// [`crate::query::DatasetClause`], or with [`DatasetView::new`] from explicit
/// graph lists.
pub struct DatasetView<'a> {
    base: &'a dyn Dataset,
    default_graphs: Vec<NamedNode>,
    named_graphs: Vec<NamedNode>,
}

impl<'a> DatasetView<'a> {
    /// Build a view from explicit `FROM` / `FROM NAMED` graph lists.
    pub fn new(
        base: &'a dyn Dataset,
        default_graphs: Vec<NamedNode>,
        named_graphs: Vec<NamedNode>,
    ) -> Self {
        Self {
            base,
            default_graphs,
            named_graphs,
        }
    }

    /// Whether the clause is empty (no `FROM`, no `FROM NAMED`); such a view is
    /// a transparent passthrough of the base dataset.
    fn is_passthrough(&self) -> bool {
        self.default_graphs.is_empty() && self.named_graphs.is_empty()
    }

    /// Triples visible in the view's active default graph: the union of the
    /// `FROM` graphs, or the base default graph when there are none.
    fn default_graph_triples(
        &self,
        pattern: &TriplePattern,
    ) -> Result<Vec<(AlgebraTerm, AlgebraTerm, AlgebraTerm)>> {
        if self.default_graphs.is_empty() {
            return self.base.find_triples(pattern);
        }
        let mut seen: HashSet<(AlgebraTerm, AlgebraTerm, AlgebraTerm)> = HashSet::new();
        let mut out: Vec<(AlgebraTerm, AlgebraTerm, AlgebraTerm)> = Vec::new();
        for graph in &self.default_graphs {
            let selector = GraphSelector::Named(graph.clone());
            for triple in self.base.find_triples_in(&selector, pattern)? {
                if seen.insert(triple.clone()) {
                    out.push(triple);
                }
            }
        }
        Ok(out)
    }
}

/// Build a [`DatasetView`] over `base` from a parsed `FROM` / `FROM NAMED`
/// clause.
///
/// The fuseki wave calls this with `Query.dataset` to materialize the RDF
/// dataset a query runs against. An empty clause yields a passthrough view.
pub fn with_dataset_clause<'a>(
    base: &'a dyn Dataset,
    clause: &crate::query::DatasetClause,
) -> DatasetView<'a> {
    DatasetView::new(
        base,
        clause.default_graphs.clone(),
        clause.named_graphs.clone(),
    )
}

impl<'a> Dataset for DatasetView<'a> {
    fn find_triples(
        &self,
        pattern: &TriplePattern,
    ) -> Result<Vec<(AlgebraTerm, AlgebraTerm, AlgebraTerm)>> {
        if self.is_passthrough() {
            return self.base.find_triples(pattern);
        }
        self.default_graph_triples(pattern)
    }

    fn find_triples_in(
        &self,
        selector: &GraphSelector,
        pattern: &TriplePattern,
    ) -> Result<Vec<(AlgebraTerm, AlgebraTerm, AlgebraTerm)>> {
        if self.is_passthrough() {
            return self.base.find_triples_in(selector, pattern);
        }
        match selector {
            GraphSelector::DefaultGraph => self.default_graph_triples(pattern),
            GraphSelector::Named(graph) => {
                // Only graphs named in FROM NAMED are visible.
                if self.named_graphs.iter().any(|n| n == graph) {
                    self.base.find_triples_in(selector, pattern)
                } else {
                    Ok(Vec::new())
                }
            }
        }
    }

    fn has_graph_support(&self) -> bool {
        self.base.has_graph_support()
    }

    fn named_graphs(&self) -> Result<Vec<AlgebraTerm>> {
        if self.is_passthrough() {
            return self.base.named_graphs();
        }
        Ok(self
            .named_graphs
            .iter()
            .cloned()
            .map(AlgebraTerm::Iri)
            .collect())
    }

    fn contains_triple(
        &self,
        subject: &AlgebraTerm,
        predicate: &AlgebraTerm,
        object: &AlgebraTerm,
    ) -> Result<bool> {
        let pattern = TriplePattern::new(subject.clone(), predicate.clone(), object.clone());
        Ok(!self.find_triples(&pattern)?.is_empty())
    }

    fn subjects(&self) -> Result<Vec<AlgebraTerm>> {
        if self.is_passthrough() {
            return self.base.subjects();
        }
        let triples = self.default_graph_triples(&wildcard_pattern())?;
        let set: HashSet<_> = triples.into_iter().map(|(s, _, _)| s).collect();
        Ok(set.into_iter().collect())
    }

    fn predicates(&self) -> Result<Vec<AlgebraTerm>> {
        if self.is_passthrough() {
            return self.base.predicates();
        }
        let triples = self.default_graph_triples(&wildcard_pattern())?;
        let set: HashSet<_> = triples.into_iter().map(|(_, p, _)| p).collect();
        Ok(set.into_iter().collect())
    }

    fn objects(&self) -> Result<Vec<AlgebraTerm>> {
        if self.is_passthrough() {
            return self.base.objects();
        }
        let triples = self.default_graph_triples(&wildcard_pattern())?;
        let set: HashSet<_> = triples.into_iter().map(|(_, _, o)| o).collect();
        Ok(set.into_iter().collect())
    }
}

#[cfg(test)]
mod store_ref_dataset_tests {
    use super::*;
    use crate::algebra::{Literal, Variable};
    use oxirs_core::model::{GraphName, Literal as CoreLiteral, NamedNode, Quad};
    use oxirs_core::rdf_store::{ConcreteStore, Store};

    #[test]
    fn store_ref_dataset_matches_typed_literal() {
        let store = ConcreteStore::new().expect("store");
        let s = NamedNode::new_unchecked("http://ex/s");
        let age = NamedNode::new_unchecked("http://ex/age");
        let xsd_int = NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#integer");

        // One typed and one plain literal with the same lexical form.
        store
            .insert(&Quad::new(
                s.clone(),
                age.clone(),
                CoreLiteral::new_typed("25", xsd_int.clone()),
                GraphName::DefaultGraph,
            ))
            .expect("insert typed");
        store
            .insert(&Quad::new(
                s.clone(),
                age.clone(),
                CoreLiteral::new("25"),
                GraphName::DefaultGraph,
            ))
            .expect("insert plain");

        let ds = StoreRefDataset::new(&store);
        let pattern = TriplePattern {
            subject: AlgebraTerm::Variable(Variable::new_unchecked("s")),
            predicate: AlgebraTerm::Iri(age.clone()),
            object: AlgebraTerm::Literal(Literal {
                value: "25".to_string(),
                language: None,
                datatype: Some(xsd_int.clone()),
            }),
        };
        let results = ds.find_triples(&pattern).expect("find");
        assert_eq!(
            results.len(),
            1,
            "typed-literal pattern must match exactly the typed triple"
        );
        // The returned object literal must preserve the integer datatype (the
        // pre-fix code coerced everything to plain xsd:string and dropped it).
        match &results[0].2 {
            AlgebraTerm::Literal(lit) => {
                assert_eq!(lit.value, "25");
                assert_eq!(
                    lit.datatype.as_ref().map(|d| d.as_str()),
                    Some("http://www.w3.org/2001/XMLSchema#integer")
                );
            }
            other => panic!("expected literal, got {other:?}"),
        }
    }

    #[test]
    fn store_ref_dataset_preserves_language_tag() {
        let store = ConcreteStore::new().expect("store");
        let s = NamedNode::new_unchecked("http://ex/s");
        let label = NamedNode::new_unchecked("http://ex/label");
        store
            .insert(&Quad::new(
                s.clone(),
                label.clone(),
                CoreLiteral::new_language_tagged_literal("hi", "en").expect("lang literal"),
                GraphName::DefaultGraph,
            ))
            .expect("insert");

        let ds = StoreRefDataset::new(&store);
        let pattern = TriplePattern {
            subject: AlgebraTerm::Variable(Variable::new_unchecked("s")),
            predicate: AlgebraTerm::Iri(label.clone()),
            object: AlgebraTerm::Literal(Literal {
                value: "hi".to_string(),
                language: Some("en".to_string()),
                datatype: None,
            }),
        };
        let results = ds.find_triples(&pattern).expect("find");
        assert_eq!(results.len(), 1, "language-tagged pattern must match");
        match &results[0].2 {
            AlgebraTerm::Literal(lit) => {
                assert_eq!(lit.language.as_deref(), Some("en"));
            }
            other => panic!("expected literal, got {other:?}"),
        }
    }
}

#[cfg(test)]
mod graph_scoping_tests {
    use super::*;
    use crate::algebra::Variable;
    use oxirs_core::model::{GraphName, NamedNode, Quad};
    use oxirs_core::rdf_store::{ConcreteStore, Store};

    fn nn(s: &str) -> NamedNode {
        NamedNode::new_unchecked(s)
    }

    fn wildcard() -> TriplePattern {
        TriplePattern {
            subject: AlgebraTerm::Variable(Variable::new_unchecked("s")),
            predicate: AlgebraTerm::Variable(Variable::new_unchecked("p")),
            object: AlgebraTerm::Variable(Variable::new_unchecked("o")),
        }
    }

    /// Store with one default-graph triple and two triples spread over two
    /// named graphs `<g1>` (two triples) and `<g2>` (one triple).
    fn seeded_store() -> ConcreteStore {
        let store = ConcreteStore::new().expect("store");
        let p = nn("http://ex/p");
        store
            .insert(&Quad::new(
                nn("http://ex/sd"),
                p.clone(),
                nn("http://ex/od"),
                GraphName::DefaultGraph,
            ))
            .expect("insert default");
        store
            .insert(&Quad::new(
                nn("http://ex/s1"),
                p.clone(),
                nn("http://ex/o1"),
                GraphName::NamedNode(nn("http://ex/g1")),
            ))
            .expect("insert g1 a");
        store
            .insert(&Quad::new(
                nn("http://ex/s2"),
                p.clone(),
                nn("http://ex/o2"),
                GraphName::NamedNode(nn("http://ex/g1")),
            ))
            .expect("insert g1 b");
        store
            .insert(&Quad::new(
                nn("http://ex/s3"),
                p,
                nn("http://ex/o3"),
                GraphName::NamedNode(nn("http://ex/g2")),
            ))
            .expect("insert g2");
        store
    }

    #[test]
    fn store_ref_find_triples_reads_default_graph_only() {
        let store = seeded_store();
        let ds = StoreRefDataset::new(&store);
        // Regression for the union bug: the plain lookup must see ONLY the one
        // default-graph triple, not all four.
        let all = ds.find_triples(&wildcard()).expect("find default");
        assert_eq!(
            all.len(),
            1,
            "plain find_triples must be default-graph-only"
        );
        assert_eq!(all[0].0, AlgebraTerm::Iri(nn("http://ex/sd")));
    }

    #[test]
    fn store_ref_find_triples_in_named_graph() {
        let store = seeded_store();
        let ds = StoreRefDataset::new(&store);
        let g1 = ds
            .find_triples_in(&GraphSelector::Named(nn("http://ex/g1")), &wildcard())
            .expect("find g1");
        assert_eq!(g1.len(), 2, "named graph g1 has two triples");
        let g2 = ds
            .find_triples_in(&GraphSelector::Named(nn("http://ex/g2")), &wildcard())
            .expect("find g2");
        assert_eq!(g2.len(), 1, "named graph g2 has one triple");
        let dg = ds
            .find_triples_in(&GraphSelector::DefaultGraph, &wildcard())
            .expect("find default");
        assert_eq!(dg.len(), 1, "default graph has one triple");
    }

    #[test]
    fn store_ref_named_graphs_enumerates() {
        let store = seeded_store();
        let ds = StoreRefDataset::new(&store);
        assert!(ds.has_graph_support());
        let mut graphs = ds.named_graphs().expect("named graphs");
        graphs.sort_by_key(|t| format!("{t:?}"));
        assert_eq!(graphs.len(), 2, "two distinct named graphs");
        assert!(graphs.contains(&AlgebraTerm::Iri(nn("http://ex/g1"))));
        assert!(graphs.contains(&AlgebraTerm::Iri(nn("http://ex/g2"))));
    }

    #[test]
    fn default_impl_find_triples_in_fails_loud() {
        // A dataset that does not override the graph methods must FAIL LOUD
        // rather than silently union all graphs.
        struct NoGraphDataset;
        impl Dataset for NoGraphDataset {
            fn find_triples(
                &self,
                _pattern: &TriplePattern,
            ) -> Result<Vec<(AlgebraTerm, AlgebraTerm, AlgebraTerm)>> {
                Ok(Vec::new())
            }
            fn contains_triple(
                &self,
                _s: &AlgebraTerm,
                _p: &AlgebraTerm,
                _o: &AlgebraTerm,
            ) -> Result<bool> {
                Ok(false)
            }
            fn subjects(&self) -> Result<Vec<AlgebraTerm>> {
                Ok(Vec::new())
            }
            fn predicates(&self) -> Result<Vec<AlgebraTerm>> {
                Ok(Vec::new())
            }
            fn objects(&self) -> Result<Vec<AlgebraTerm>> {
                Ok(Vec::new())
            }
        }
        let ds = NoGraphDataset;
        assert!(!ds.has_graph_support());
        assert!(ds
            .find_triples_in(&GraphSelector::Named(nn("http://ex/g")), &wildcard())
            .is_err());
        assert!(ds.named_graphs().is_err());
    }

    #[test]
    fn dataset_view_from_single_graph_default() {
        let store = seeded_store();
        let base = StoreRefDataset::new(&store);
        // FROM <g1>: default graph becomes g1's two triples.
        let view = DatasetView::new(&base, vec![nn("http://ex/g1")], vec![]);
        let default = view.find_triples(&wildcard()).expect("view default");
        assert_eq!(default.len(), 2, "FROM <g1> default graph = g1's triples");
        // FROM-only clause exposes no named graphs.
        assert!(view.named_graphs().expect("named").is_empty());
    }

    #[test]
    fn dataset_view_from_multiple_graphs_union() {
        let store = seeded_store();
        let base = StoreRefDataset::new(&store);
        // FROM <g1> FROM <g2>: default graph = union (2 + 1 = 3 triples).
        let view = DatasetView::new(&base, vec![nn("http://ex/g1"), nn("http://ex/g2")], vec![]);
        let default = view.find_triples(&wildcard()).expect("view default");
        assert_eq!(default.len(), 3, "FROM union of g1+g2 = three triples");
    }

    #[test]
    fn dataset_view_from_named_restricts_visibility() {
        let store = seeded_store();
        let base = StoreRefDataset::new(&store);
        // FROM NAMED <g1> only: g1 visible, g2 not.
        let view = DatasetView::new(&base, vec![], vec![nn("http://ex/g1")]);
        let named = view.named_graphs().expect("named graphs");
        assert_eq!(named, vec![AlgebraTerm::Iri(nn("http://ex/g1"))]);
        // g1 is visible.
        let g1 = view
            .find_triples_in(&GraphSelector::Named(nn("http://ex/g1")), &wildcard())
            .expect("find g1");
        assert_eq!(g1.len(), 2);
        // g2 is NOT in FROM NAMED -> invisible -> empty.
        let g2 = view
            .find_triples_in(&GraphSelector::Named(nn("http://ex/g2")), &wildcard())
            .expect("find g2");
        assert!(g2.is_empty(), "graph outside FROM NAMED must be invisible");
        // No FROM -> default graph falls back to the store default (one triple).
        let default = view.find_triples(&wildcard()).expect("view default");
        assert_eq!(default.len(), 1, "no FROM => store default graph");
    }

    #[test]
    fn dataset_view_empty_clause_is_passthrough() {
        let store = seeded_store();
        let base = StoreRefDataset::new(&store);
        let view = DatasetView::new(&base, vec![], vec![]);
        // Passthrough: identical to the base dataset's behavior.
        let default = view.find_triples(&wildcard()).expect("view default");
        assert_eq!(
            default.len(),
            1,
            "empty clause preserves default-graph read"
        );
        let mut graphs = view.named_graphs().expect("named");
        graphs.sort_by_key(|t| format!("{t:?}"));
        assert_eq!(graphs.len(), 2, "empty clause preserves base named graphs");
    }

    #[test]
    fn graph_scoped_dataset_routes_plain_lookup() {
        let store = seeded_store();
        let base = StoreRefDataset::new(&store);
        // Wrapping in a scope for g1 makes the *plain* find_triples read g1.
        let scoped = GraphScopedDataset::new(&base, GraphSelector::Named(nn("http://ex/g1")));
        let triples = scoped.find_triples(&wildcard()).expect("scoped find");
        assert_eq!(triples.len(), 2, "scoped plain lookup reads g1 only");
    }
}
