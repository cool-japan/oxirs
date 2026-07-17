//! Streaming query iterators over the triple store (F5).
//!
//! [`TdbStore::query_triples`](crate::store::TdbStore::query_triples)
//! materializes and *caches* the full result of a pattern scan. That is fine
//! for small, repeated lookups but is exactly the wrong shape for a SPARQL
//! engine binding, which wants to pull decoded triples lazily and stop early.
//!
//! This module adds a lazy, streaming path:
//! [`TdbStore::stream_triples`] returns a [`TripleTermIter`] that pulls one
//! [`Triple`] at a time from the selected B+Tree (via
//! [`TripleIndexes::scan`](crate::index::TripleIndexes::scan)) and decodes it
//! against the dictionary on the fly — never buffering the whole result set,
//! and never touching the query cache. [`TdbStore::for_each_triple`] is the
//! callback equivalent.

use crate::dictionary::{Dictionary, Term};
use crate::error::{Result, TdbError};
use crate::index::{Triple, TripleScan};
use crate::store::store_impl::TdbStore;

/// Decode a node-encoded [`Triple`] into its `(subject, predicate, object)`
/// [`Term`]s, failing loudly if any id is absent from the dictionary (which
/// would indicate index/dictionary corruption rather than an empty result).
pub(crate) fn decode_triple_terms(
    dictionary: &Dictionary,
    triple: Triple,
) -> Result<(Term, Term, Term)> {
    let subject = dictionary
        .decode(triple.subject)?
        .ok_or_else(|| TdbError::Other("Subject id not found in dictionary".to_string()))?;
    let predicate = dictionary
        .decode(triple.predicate)?
        .ok_or_else(|| TdbError::Other("Predicate id not found in dictionary".to_string()))?;
    let object = dictionary
        .decode(triple.object)?
        .ok_or_else(|| TdbError::Other("Object id not found in dictionary".to_string()))?;
    Ok((subject, predicate, object))
}

/// A lazy, streaming iterator over decoded default-graph triples.
///
/// Created by [`TdbStore::stream_triples`]. Each `next()` pulls one node-encoded
/// triple from the underlying [`TripleScan`] and decodes it against the
/// dictionary; the full result set is never materialized in memory at once
/// (only the B+Tree's current leaf page is buffered by the scan).
pub struct TripleTermIter<'a> {
    /// The node-level scan, or `None` for an empty result (e.g. a pattern that
    /// references a term absent from the dictionary).
    scan: Option<TripleScan>,
    /// Dictionary used to decode node ids back into terms.
    dictionary: &'a Dictionary,
}

impl<'a> TripleTermIter<'a> {
    /// Wrap a node-level scan with on-the-fly term decoding.
    pub(crate) fn new(scan: TripleScan, dictionary: &'a Dictionary) -> Self {
        Self {
            scan: Some(scan),
            dictionary,
        }
    }

    /// An iterator that yields nothing (used when a pattern term is unknown).
    pub(crate) fn empty(dictionary: &'a Dictionary) -> Self {
        Self {
            scan: None,
            dictionary,
        }
    }
}

impl Iterator for TripleTermIter<'_> {
    type Item = Result<(Term, Term, Term)>;

    fn next(&mut self) -> Option<Self::Item> {
        let scan = self.scan.as_mut()?;
        match scan.next()? {
            Ok(triple) => Some(decode_triple_terms(self.dictionary, triple)),
            Err(e) => Some(Err(e)),
        }
    }
}

impl TdbStore {
    /// Open a lazy, streaming iterator over the default-graph triples matching
    /// `(subject, predicate, object)` (`None` = wildcard).
    ///
    /// Unlike [`TdbStore::query_triples`], this does not cache or fully
    /// materialize the result: decoded triples are produced one at a time as
    /// the iterator is advanced, which is what a SPARQL engine binding needs.
    /// A pattern component whose term is not in the dictionary yields an empty
    /// iterator.
    pub fn stream_triples(
        &self,
        subject: Option<&Term>,
        predicate: Option<&Term>,
        object: Option<&Term>,
    ) -> Result<TripleTermIter<'_>> {
        let s_id = match subject {
            None => None,
            Some(term) => match self.dictionary.lookup(term)? {
                Some(id) => Some(id),
                None => return Ok(TripleTermIter::empty(&self.dictionary)),
            },
        };
        let p_id = match predicate {
            None => None,
            Some(term) => match self.dictionary.lookup(term)? {
                Some(id) => Some(id),
                None => return Ok(TripleTermIter::empty(&self.dictionary)),
            },
        };
        let o_id = match object {
            None => None,
            Some(term) => match self.dictionary.lookup(term)? {
                Some(id) => Some(id),
                None => return Ok(TripleTermIter::empty(&self.dictionary)),
            },
        };

        let scan = self.indexes.scan(s_id, p_id, o_id)?;
        Ok(TripleTermIter::new(scan, &self.dictionary))
    }

    /// Invoke `f` for each default-graph triple matching the pattern, streaming
    /// (never materializing the whole result). Stops early and propagates the
    /// error if `f` returns `Err`.
    pub fn for_each_triple<F>(
        &self,
        subject: Option<&Term>,
        predicate: Option<&Term>,
        object: Option<&Term>,
        mut f: F,
    ) -> Result<()>
    where
        F: FnMut((Term, Term, Term)) -> Result<()>,
    {
        for item in self.stream_triples(subject, predicate, object)? {
            f(item?)?;
        }
        Ok(())
    }
}
