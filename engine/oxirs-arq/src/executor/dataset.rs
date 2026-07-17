//! Dataset Implementation
//!
//! This module provides dataset abstractions and implementations for query execution.

use crate::algebra::{PropertyPath, Term as AlgebraTerm, TriplePattern};
use crate::path::{PathDataset, PropertyPath as PathPropertyPath};
use anyhow::{anyhow, Result};
use oxirs_core::RdfTerm;
use std::collections::HashSet;

/// Dataset trait for data access during query execution
pub trait Dataset: Send + Sync {
    /// Find all triples matching the given pattern
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
}

/// In-memory dataset implementation for testing
#[derive(Debug, Clone)]
pub struct InMemoryDataset {
    triples: Vec<(AlgebraTerm, AlgebraTerm, AlgebraTerm)>,
}

impl InMemoryDataset {
    pub fn new() -> Self {
        Self {
            triples: Vec::new(),
        }
    }

    pub fn add_triple(
        &mut self,
        subject: AlgebraTerm,
        predicate: AlgebraTerm,
        object: AlgebraTerm,
    ) {
        self.triples.push((subject, predicate, object));
    }

    pub fn from_triples(triples: Vec<(AlgebraTerm, AlgebraTerm, AlgebraTerm)>) -> Self {
        Self { triples }
    }
}

impl Dataset for InMemoryDataset {
    fn find_triples(
        &self,
        pattern: &TriplePattern,
    ) -> Result<Vec<(AlgebraTerm, AlgebraTerm, AlgebraTerm)>> {
        let results = self
            .triples
            .iter()
            .filter(|(s, p, o)| {
                matches_term(&pattern.subject, s)
                    && matches_term(&pattern.predicate, p)
                    && matches_term(&pattern.object, o)
            })
            .cloned()
            .collect();
        Ok(results)
    }

    fn contains_triple(
        &self,
        subject: &AlgebraTerm,
        predicate: &AlgebraTerm,
        object: &AlgebraTerm,
    ) -> Result<bool> {
        Ok(self
            .triples
            .iter()
            .any(|(s, p, o)| s == subject && p == predicate && o == object))
    }

    fn subjects(&self) -> Result<Vec<AlgebraTerm>> {
        let subjects: HashSet<_> = self.triples.iter().map(|(s, _, _)| s.clone()).collect();
        Ok(subjects.into_iter().collect())
    }

    fn predicates(&self) -> Result<Vec<AlgebraTerm>> {
        let predicates: HashSet<_> = self.triples.iter().map(|(_, p, _)| p.clone()).collect();
        Ok(predicates.into_iter().collect())
    }

    fn objects(&self) -> Result<Vec<AlgebraTerm>> {
        let objects: HashSet<_> = self.triples.iter().map(|(_, _, o)| o.clone()).collect();
        Ok(objects.into_iter().collect())
    }
}

impl Default for InMemoryDataset {
    fn default() -> Self {
        Self::new()
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
        use oxirs_core::rdf_store::Store;

        let (subject, predicate, object) = pattern_to_query_terms(pattern)?;
        let quads = self.store.find_quads(
            subject.as_ref(),
            predicate.as_ref(),
            object.as_ref(),
            None, // default graph
        )?;
        Ok(quads_to_algebra_triples(quads))
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
        let (subject, predicate, object) = pattern_to_query_terms(pattern)?;
        let quads =
            self.store
                .find_quads(subject.as_ref(), predicate.as_ref(), object.as_ref(), None)?;
        Ok(quads_to_algebra_triples(quads))
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
