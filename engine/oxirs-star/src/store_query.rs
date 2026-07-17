//! Query and iteration operations for the RDF-star store.
//!
//! This sibling module of `crate::store` hosts pattern queries, iteration
//! helpers (including [`StreamingTripleIterator`]), conversions from core
//! RDF types back into [`StarTerm`]/[`StarTriple`] values, and the
//! graph-import/export helpers.

use std::collections::BTreeSet;

use oxirs_core::rdf_store::Store;
use tracing::{info, span, Level};

use crate::model::{StarGraph, StarTerm, StarTriple};
use crate::store::conversion;
use crate::store::StarStore;
use crate::{StarError, StarResult};

impl StarStore {
    /// Query triples matching a pattern.
    ///
    /// S/P/O-bound patterns that contain no quoted-triple term are delegated
    /// to the core store's indexed `find_quads` lookup instead of scanning
    /// (and converting) every triple in the store; only quoted-triple
    /// patterns fall back to a full scan of the (typically much smaller)
    /// star-triple set.
    pub fn query(
        &self,
        subject: Option<&StarTerm>,
        predicate: Option<&StarTerm>,
        object: Option<&StarTerm>,
    ) -> StarResult<Vec<StarTriple>> {
        let mut results = Vec::new();

        // Query star triples (those containing quoted triples). This set is
        // kept separate from the (indexed) core store, so it is always a
        // linear scan; it is expected to be small relative to regular data.
        {
            let star_triples = self.star_triples.read().unwrap_or_else(|e| e.into_inner());
            for triple in star_triples.iter() {
                let matches = (subject.is_none() || subject == Some(&triple.subject))
                    && (predicate.is_none() || predicate == Some(&triple.predicate))
                    && (object.is_none() || object == Some(&triple.object));

                if matches {
                    results.push(triple.clone());
                }
            }
        }

        // Query regular triples from core store if no quoted triples appear
        // in the pattern. Delegate directly to the core store's indexed
        // find_quads (via query_core_store) instead of dumping and
        // re-filtering the entire store.
        if subject.map_or(true, |s| !matches!(s, StarTerm::QuotedTriple(_)))
            && predicate.map_or(true, |p| !matches!(p, StarTerm::QuotedTriple(_)))
            && object.map_or(true, |o| !matches!(o, StarTerm::QuotedTriple(_)))
        {
            let mut seen: std::collections::HashSet<StarTriple> = results.iter().cloned().collect();
            let core_results = self.query_core_store(subject, predicate, object)?;
            for triple in core_results {
                // query_core_store only ever returns non-quoted triples
                // (quoted triples live in star_triples), but guard anyway.
                if !triple.contains_quoted_triples() && seen.insert(triple.clone()) {
                    results.push(triple);
                }
            }
        }

        Ok(results)
    }

    /// Get all triples in the store.
    ///
    /// This performs a genuine full scan of both the star-triple set and the
    /// core store, and is intended for full-scan use cases (export, stats,
    /// iteration). Bounded S/P/O lookups should use [`StarStore::query`]
    /// instead, which delegates to the core store's indices.
    pub fn triples(&self) -> Vec<StarTriple> {
        let mut all_triples = Vec::new();

        // Add star triples (clone to release lock quickly)
        {
            let star_triples = self.star_triples.read().unwrap_or_else(|e| e.into_inner());
            all_triples.extend(star_triples.clone());
        }

        // Add regular triples from core store (release lock quickly)
        {
            let core_store = self.core_store.read().unwrap_or_else(|e| e.into_inner());
            if let Ok(quads) = core_store.find_quads(None, None, None, None) {
                drop(core_store); // Release lock before conversion
                all_triples.extend(self.convert_core_quads_lossy(quads, "triples"));
            }
        }

        all_triples
    }

    /// Find triples that contain a specific quoted triple
    pub fn find_triples_containing_quoted(&self, quoted_triple: &StarTriple) -> Vec<StarTriple> {
        let span = span!(Level::DEBUG, "find_triples_containing_quoted");
        let _enter = span.enter();

        let key = self.quoted_triple_key(quoted_triple);
        let index = self
            .quoted_triple_index
            .read()
            .unwrap_or_else(|e| e.into_inner());
        let star_triples = self.star_triples.read().unwrap_or_else(|e| e.into_inner());

        if let Some(indices) = index.signature_to_indices.get(&key) {
            indices
                .iter()
                .filter_map(|&idx| star_triples.get(idx))
                .cloned()
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Advanced query method: find triples by quoted triple pattern
    pub fn find_triples_by_quoted_pattern(
        &self,
        subject_pattern: Option<&StarTerm>,
        predicate_pattern: Option<&StarTerm>,
        object_pattern: Option<&StarTerm>,
    ) -> Vec<StarTriple> {
        let span = span!(Level::DEBUG, "find_triples_by_quoted_pattern");
        let _enter = span.enter();

        let index = self
            .quoted_triple_index
            .read()
            .unwrap_or_else(|e| e.into_inner());
        let star_triples = self.star_triples.read().unwrap_or_else(|e| e.into_inner());
        let mut candidate_indices: Option<BTreeSet<usize>> = None;

        // Use subject index if subject pattern is provided
        if let Some(subject_term) = subject_pattern {
            let mut found_indices = BTreeSet::new();

            // Search in all index types for the subject term, as it could appear in any position within quoted triples
            let subject_key = format!("SUBJ:{subject_term}");
            if let Some(indices) = index.subject_index.get(&subject_key) {
                found_indices.extend(indices);
            }

            let predicate_key = format!("PRED:{subject_term}");
            if let Some(indices) = index.predicate_index.get(&predicate_key) {
                found_indices.extend(indices);
            }

            let object_key = format!("OBJ:{subject_term}");
            if let Some(indices) = index.object_index.get(&object_key) {
                found_indices.extend(indices);
            }

            if found_indices.is_empty() {
                return Vec::new(); // No matches
            }

            candidate_indices = Some(found_indices);
        }

        // Use predicate index if predicate pattern is provided
        if let Some(predicate_term) = predicate_pattern {
            let predicate_key = format!("PRED:{predicate_term}");
            if let Some(indices) = index.predicate_index.get(&predicate_key) {
                if let Some(ref mut candidates) = candidate_indices {
                    *candidates = candidates.intersection(indices).cloned().collect();
                } else {
                    candidate_indices = Some(indices.clone());
                }

                if candidate_indices
                    .as_ref()
                    .expect("candidate_indices should be Some after setting")
                    .is_empty()
                {
                    return Vec::new(); // No matches
                }
            } else {
                return Vec::new(); // No matches
            }
        }

        // Use object index if object pattern is provided
        if let Some(object_term) = object_pattern {
            let object_key = format!("OBJ:{object_term}");
            if let Some(indices) = index.object_index.get(&object_key) {
                if let Some(ref mut candidates) = candidate_indices {
                    *candidates = candidates.intersection(indices).cloned().collect();
                } else {
                    candidate_indices = Some(indices.clone());
                }

                if candidate_indices
                    .as_ref()
                    .expect("candidate_indices should be Some after setting")
                    .is_empty()
                {
                    return Vec::new(); // No matches
                }
            } else {
                return Vec::new(); // No matches
            }
        }

        // If no pattern was provided, return all triples with quoted triples
        let final_indices = candidate_indices.unwrap_or_else(|| {
            index
                .signature_to_indices
                .values()
                .flat_map(|indices| indices.iter())
                .cloned()
                .collect()
        });

        final_indices
            .iter()
            .filter_map(|&idx| star_triples.get(idx))
            .cloned()
            .collect()
    }

    /// Find triples by nesting depth
    pub fn find_triples_by_nesting_depth(
        &self,
        min_depth: usize,
        max_depth: Option<usize>,
    ) -> Vec<StarTriple> {
        let span = span!(Level::DEBUG, "find_triples_by_nesting_depth");
        let _enter = span.enter();

        let mut results = Vec::new();
        let max_d = max_depth.unwrap_or(usize::MAX);

        // If we're looking for depth 0 triples, include regular triples from core_store
        if min_depth == 0 {
            let core_store = self.core_store.read().unwrap_or_else(|e| e.into_inner());
            if let Ok(quads) = core_store.find_quads(None, None, None, None) {
                drop(core_store);
                for star_triple in
                    self.convert_core_quads_lossy(quads, "find_triples_by_nesting_depth")
                {
                    if !star_triple.contains_quoted_triples() {
                        results.push(star_triple);
                    }
                }
            }
        }

        // Find star triples (quoted triples) by nesting depth
        let index = self
            .quoted_triple_index
            .read()
            .unwrap_or_else(|e| e.into_inner());
        let star_triples = self.star_triples.read().unwrap_or_else(|e| e.into_inner());
        let mut result_indices = BTreeSet::new();

        for (&_depth, indices) in index.nesting_depth_index.range(min_depth..=max_d) {
            result_indices.extend(indices);
        }

        results.extend(
            result_indices
                .iter()
                .filter_map(|&idx: &usize| star_triples.get(idx))
                .cloned(),
        );

        results
    }

    /// Export the store as a StarGraph
    pub fn to_graph(&self) -> StarGraph {
        let star_triples = self.star_triples.read().unwrap_or_else(|e| e.into_inner());
        let mut graph = StarGraph::new();

        // Add star triples (containing quoted triples)
        for triple in star_triples.iter() {
            // Safe because we validate triples on insert
            graph
                .insert(triple.clone())
                .expect("triple should be valid after validation on insert");
        }

        // Add regular triples from core store
        let core_store = self.core_store.read().unwrap_or_else(|e| e.into_inner());
        if let Ok(quads) = core_store.find_quads(None, None, None, None) {
            drop(core_store);
            for star_triple in self.convert_core_quads_lossy(quads, "to_graph") {
                // Only add if it doesn't contain quoted triples (those are already in star_triples)
                if !star_triple.contains_quoted_triples() {
                    graph
                        .insert(star_triple)
                        .expect("triple should be valid after core store validation");
                }
            }
        }

        graph
    }

    /// Import triples from a StarGraph
    pub fn from_graph(&self, graph: &StarGraph) -> StarResult<()> {
        let span = span!(Level::INFO, "import_from_graph");
        let _enter = span.enter();

        for triple in graph.triples() {
            self.insert(triple)?;
        }

        info!("Imported {} triples from graph", graph.len());
        Ok(())
    }

    /// Query triples from both core store and star store
    pub fn query_triples(
        &self,
        subject: Option<&StarTerm>,
        predicate: Option<&StarTerm>,
        object: Option<&StarTerm>,
    ) -> StarResult<Vec<StarTriple>> {
        let mut results = Vec::new();

        // Query star triples
        let star_triples = self.star_triples.read().unwrap_or_else(|e| e.into_inner());
        for triple in star_triples.iter() {
            if self.triple_matches(triple, subject, predicate, object) {
                results.push(triple.clone());
            }
        }

        // If no quoted triple patterns, also query core store
        let has_quoted_pattern = [subject, predicate, object]
            .iter()
            .any(|term| term.is_some_and(|t| t.is_quoted_triple()));

        if !has_quoted_pattern {
            // Convert patterns to core RDF terms and query core store
            let core_results = self.query_core_store(subject, predicate, object)?;
            results.extend(core_results);
        }

        Ok(results)
    }

    /// Query the core store with converted patterns
    pub(crate) fn query_core_store(
        &self,
        subject: Option<&StarTerm>,
        predicate: Option<&StarTerm>,
        object: Option<&StarTerm>,
    ) -> StarResult<Vec<StarTriple>> {
        let core_store = self.core_store.read().unwrap_or_else(|e| e.into_inner());

        // Convert patterns to core types
        let core_subject = match subject {
            Some(term) => Some(conversion::star_term_to_subject(term)?),
            None => None,
        };

        let core_predicate = match predicate {
            Some(term) => Some(conversion::star_term_to_predicate(term)?),
            None => None,
        };

        let core_object = match object {
            Some(term) => Some(conversion::star_term_to_object(term)?),
            None => None,
        };

        // Query core store (find quads and convert to triples)
        let core_quads = core_store
            .find_quads(
                core_subject.as_ref(),
                core_predicate.as_ref(),
                core_object.as_ref(),
                None, // Query all graphs
            )
            .map_err(StarError::CoreError)?;

        // Convert results back to StarTriples
        let mut results = Vec::new();
        for quad in core_quads {
            // Convert quad to triple (lose graph information)
            let triple = oxirs_core::model::Triple::new(
                quad.subject().clone(),
                quad.predicate().clone(),
                quad.object().clone(),
            );
            let star_triple = self.convert_from_core_triple(&triple)?;
            results.push(star_triple);
        }

        Ok(results)
    }

    /// Convert core quads (`quads`) into [`StarTriple`]s, logging a single
    /// `warn!` with the skipped count (rather than silently discarding data)
    /// if any triple could not be converted.
    ///
    /// This currently affects quoted triples synced into the core store via
    /// [`oxirs_core::model::Subject::QuotedTriple`] /
    /// [`oxirs_core::model::Object::QuotedTriple`]: `StarStore`'s
    /// core-to-star conversion does not support them yet (see
    /// [`StarStore::convert_subject_from_core`] /
    /// [`StarStore::convert_object_from_core`]), so any such quad is
    /// skipped here rather than surfacing an error to every caller of the
    /// read APIs that go through this helper.
    pub(crate) fn convert_core_quads_lossy(
        &self,
        quads: Vec<oxirs_core::model::Quad>,
        context: &str,
    ) -> Vec<StarTriple> {
        let mut triples = Vec::with_capacity(quads.len());
        let mut skipped = 0usize;

        for quad in quads {
            let core_triple = quad.to_triple();
            match self.convert_from_core_triple(&core_triple) {
                Ok(star_triple) => triples.push(star_triple),
                Err(_) => skipped += 1,
            }
        }

        if skipped > 0 {
            tracing::warn!(
                skipped_count = skipped,
                context,
                "Skipped {skipped} triple(s) from the core store during '{context}': quoted \
                 triples synced from the core store are not yet supported by StarStore's read \
                 APIs"
            );
        }

        triples
    }

    /// Convert a core RDF Triple to a StarTriple
    pub(crate) fn convert_from_core_triple(
        &self,
        triple: &oxirs_core::model::Triple,
    ) -> StarResult<StarTriple> {
        let subject = self.convert_subject_from_core(triple.subject())?;
        let predicate = self.convert_predicate_from_core(triple.predicate())?;
        let object = self.convert_object_from_core(triple.object())?;

        Ok(StarTriple::new(subject, predicate, object))
    }

    /// Convert core Subject to StarTerm
    pub(crate) fn convert_subject_from_core(
        &self,
        subject: &oxirs_core::model::Subject,
    ) -> StarResult<StarTerm> {
        match subject {
            oxirs_core::model::Subject::NamedNode(nn) => Ok(StarTerm::iri(nn.as_str())?),
            oxirs_core::model::Subject::BlankNode(bn) => Ok(StarTerm::blank_node(bn.as_str())?),
            oxirs_core::model::Subject::Variable(_) => Err(StarError::invalid_term_type(
                "Variables are not supported in subjects for RDF-star storage".to_string(),
            )),
            oxirs_core::model::Subject::QuotedTriple(_) => Err(StarError::invalid_term_type(
                "Quoted triples from core are not yet supported".to_string(),
            )),
        }
    }

    /// Convert core Predicate to StarTerm
    pub(crate) fn convert_predicate_from_core(
        &self,
        predicate: &oxirs_core::model::Predicate,
    ) -> StarResult<StarTerm> {
        match predicate {
            oxirs_core::model::Predicate::NamedNode(nn) => Ok(StarTerm::iri(nn.as_str())?),
            oxirs_core::model::Predicate::Variable(_) => Err(StarError::invalid_term_type(
                "Variables are not supported in predicates for RDF-star storage".to_string(),
            )),
        }
    }

    /// Convert core Object to StarTerm
    pub(crate) fn convert_object_from_core(
        &self,
        object: &oxirs_core::model::Object,
    ) -> StarResult<StarTerm> {
        match object {
            oxirs_core::model::Object::NamedNode(nn) => Ok(StarTerm::iri(nn.as_str())?),
            oxirs_core::model::Object::BlankNode(bn) => Ok(StarTerm::blank_node(bn.as_str())?),
            oxirs_core::model::Object::Literal(lit) => {
                let language = lit.language().map(|lang| lang.to_string());
                let datatype = if lit.is_lang_string() {
                    // Language-tagged literals don't need explicit datatype
                    None
                } else {
                    let dt_iri = lit.datatype().as_str();
                    // Don't include xsd:string datatype for simple literals (it's implicit)
                    if dt_iri == "http://www.w3.org/2001/XMLSchema#string" {
                        None
                    } else {
                        Some(crate::model::NamedNode {
                            iri: dt_iri.to_string(),
                        })
                    }
                };

                let star_literal = crate::model::Literal {
                    value: lit.value().to_string(),
                    language,
                    datatype,
                };
                Ok(StarTerm::Literal(star_literal))
            }
            oxirs_core::model::Object::Variable(_) => Err(StarError::invalid_term_type(
                "Variables are not supported in objects for RDF-star storage".to_string(),
            )),
            oxirs_core::model::Object::QuotedTriple(_) => Err(StarError::invalid_term_type(
                "Quoted triples from core are not yet supported".to_string(),
            )),
        }
    }

    /// Get a vector of all triples (cloned to avoid lifetime issues)
    pub fn all_triples(&self) -> Vec<StarTriple> {
        let mut all_triples = Vec::new();

        // Add star triples (containing quoted triples)
        {
            let star_triples = self.star_triples.read().unwrap_or_else(|e| e.into_inner());
            all_triples.extend(star_triples.clone());
        }

        // Add regular triples from core store
        {
            let core_store = self.core_store.read().unwrap_or_else(|e| e.into_inner());
            if let Ok(quads) = core_store.find_quads(None, None, None, None) {
                drop(core_store); // Release lock before conversion
                for star_triple in self.convert_core_quads_lossy(quads, "all_triples") {
                    // Only add if it doesn't contain quoted triples (those are already in star_triples)
                    if !star_triple.contains_quoted_triples() {
                        all_triples.push(star_triple);
                    }
                }
            }
        }

        all_triples
    }

    /// Get an iterator over all triples using a safe implementation
    pub fn iter(&self) -> impl Iterator<Item = StarTriple> + use<> {
        // Clone all triples to avoid holding the lock
        // This is safe but potentially memory-intensive for large stores
        // For production use, consider using the streaming_iter method
        self.all_triples().into_iter()
    }

    /// Get a streaming iterator that processes triples in chunks
    /// This is more memory-efficient for large stores
    pub fn streaming_iter(&self, chunk_size: usize) -> StreamingTripleIterator<'_> {
        StreamingTripleIterator::new(self, chunk_size)
    }
}

/// A memory-efficient streaming iterator for large triple stores
pub struct StreamingTripleIterator<'a> {
    store: &'a StarStore,
    chunk_size: usize,
    current_chunk: Vec<StarTriple>,
    current_index: usize,
    total_processed: usize,
}

impl<'a> StreamingTripleIterator<'a> {
    pub(crate) fn new(store: &'a StarStore, chunk_size: usize) -> Self {
        Self {
            store,
            chunk_size: chunk_size.max(1),
            current_chunk: Vec::new(),
            current_index: 0,
            total_processed: 0,
        }
    }

    fn load_next_chunk(&mut self) -> bool {
        // Get all triples (both star triples and regular triples from core store)
        let all_triples = self.store.all_triples();

        // Calculate the range for the next chunk
        let start = self.total_processed;
        let end = (start + self.chunk_size).min(all_triples.len());

        if start >= all_triples.len() {
            return false;
        }

        // Load the chunk
        self.current_chunk.clear();
        self.current_chunk
            .extend(all_triples.iter().skip(start).take(end - start).cloned());

        self.current_index = 0;
        !self.current_chunk.is_empty()
    }
}

impl<'a> Iterator for StreamingTripleIterator<'a> {
    type Item = StarTriple;

    fn next(&mut self) -> Option<Self::Item> {
        // If we've exhausted the current chunk, load the next one
        if self.current_index >= self.current_chunk.len() && !self.load_next_chunk() {
            return None;
        }

        // Return the next triple from the current chunk
        let triple = self.current_chunk.get(self.current_index).cloned();
        if triple.is_some() {
            self.current_index += 1;
            self.total_processed += 1;
        }
        triple
    }
}

#[cfg(test)]
mod tests {
    use crate::model::{StarTerm, StarTriple};
    use crate::store::StarStore;
    use crate::StarResult;

    /// Regression test for the P0 performance fix: `query()` must delegate
    /// bound S/P/O patterns to the core store's indexed `find_quads` lookup
    /// (via `query_core_store`) instead of dumping and re-scanning the whole
    /// store, and must not return duplicate results.
    #[test]
    fn test_query_bounded_pattern_uses_indexed_lookup() -> StarResult<()> {
        let store = StarStore::new();

        let alice = StarTerm::iri("http://example.org/alice")?;
        let bob = StarTerm::iri("http://example.org/bob")?;
        let knows = StarTerm::iri("http://example.org/knows")?;
        let likes = StarTerm::iri("http://example.org/likes")?;

        store.insert(&StarTriple::new(alice.clone(), knows.clone(), bob.clone()))?;
        store.insert(&StarTriple::new(alice.clone(), likes.clone(), bob.clone()))?;
        store.insert(&StarTriple::new(bob.clone(), knows.clone(), alice.clone()))?;

        // Subject-bound pattern: only alice's two triples should come back.
        let results = store.query(Some(&alice), None, None)?;
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|t| t.subject == alice));

        // Fully-bound pattern: exactly one match, no duplicates.
        let results = store.query(Some(&alice), Some(&knows), Some(&bob))?;
        assert_eq!(results.len(), 1);

        // Predicate-bound pattern spanning multiple subjects.
        let results = store.query(None, Some(&knows), None)?;
        assert_eq!(results.len(), 2);

        Ok(())
    }

    #[test]
    fn test_query_matches_star_and_regular_triples() -> StarResult<()> {
        let store = StarStore::new();

        let inner = StarTriple::new(
            StarTerm::iri("http://example.org/s1")?,
            StarTerm::iri("http://example.org/p1")?,
            StarTerm::literal("o1")?,
        );
        let quoted_triple = StarTriple::new(
            StarTerm::quoted_triple(inner),
            StarTerm::iri("http://example.org/certainty")?,
            StarTerm::literal("0.9")?,
        );
        let regular_triple = StarTriple::new(
            StarTerm::iri("http://example.org/s2")?,
            StarTerm::iri("http://example.org/p2")?,
            StarTerm::literal("o2")?,
        );

        store.insert(&quoted_triple)?;
        store.insert(&regular_triple)?;

        // Unbound pattern returns both the quoted-triple-bearing triple and
        // the plain triple, with no duplicates.
        let results = store.query(None, None, None)?;
        assert_eq!(results.len(), 2);
        assert!(results.contains(&quoted_triple));
        assert!(results.contains(&regular_triple));

        Ok(())
    }

    /// Regression test for the P2 error-handling fix: quoted triples synced
    /// into the core store (e.g. by external tooling writing directly to
    /// `oxirs_core::rdf_store`, bypassing `StarStore::insert`'s routing to
    /// `star_triples`) are not silently swallowed alongside otherwise-valid
    /// data — they are skipped individually (StarStore's core->star
    /// conversion does not support them yet) while unrelated triples in the
    /// same read are still returned correctly. Whether a `warn!` was
    /// actually emitted is not asserted here (would require a tracing
    /// test-capture dependency); this test covers the functional contract:
    /// no panic, no silent loss of *other* data.
    #[test]
    fn test_core_quoted_triples_are_skipped_not_silently_corrupting_reads() -> StarResult<()> {
        use oxirs_core::model::{
            NamedNode as CoreNamedNode, Object as CoreObject, Predicate as CorePredicate,
            Quad as CoreQuad, QuotedTriple as CoreQuotedTriple, Subject as CoreSubject,
            Triple as CoreTriple,
        };
        use oxirs_core::rdf_store::ConcreteStore;

        let store = StarStore::new();

        // A regular, fully-convertible triple.
        let regular = StarTriple::new(
            StarTerm::iri("http://example.org/s")?,
            StarTerm::iri("http://example.org/p")?,
            StarTerm::literal("o")?,
        );
        store.insert(&regular)?;

        // Directly insert a quad into the core store whose *subject* is a
        // QuotedTriple, simulating data synced in from outside StarStore's
        // own insert() path (which always routes quoted triples to
        // star_triples, never to the core store).
        let inner = CoreTriple::new(
            CoreSubject::NamedNode(
                CoreNamedNode::new("http://example.org/inner-s").expect("valid IRI"),
            ),
            CorePredicate::NamedNode(
                CoreNamedNode::new("http://example.org/inner-p").expect("valid IRI"),
            ),
            CoreObject::NamedNode(
                CoreNamedNode::new("http://example.org/inner-o").expect("valid IRI"),
            ),
        );
        let quoted_subject_quad = CoreQuad::new_default_graph(
            CoreSubject::QuotedTriple(Box::new(CoreQuotedTriple::new(inner))),
            CorePredicate::NamedNode(
                CoreNamedNode::new("http://example.org/meta").expect("valid IRI"),
            ),
            CoreObject::NamedNode(
                CoreNamedNode::new("http://example.org/unsupported").expect("valid IRI"),
            ),
        );
        {
            let core_store = store.core_store.write().unwrap_or_else(|e| e.into_inner());
            ConcreteStore::insert_quad(&core_store, quoted_subject_quad)
                .expect("core-level insert should succeed");
        }

        // The regular triple is still returned correctly; the unsupported
        // quoted-subject quad is skipped rather than panicking or silently
        // discarding the *other* data too.
        let all = store.triples();
        assert!(all.contains(&regular));
        assert_eq!(all.len(), 1);

        Ok(())
    }
}
