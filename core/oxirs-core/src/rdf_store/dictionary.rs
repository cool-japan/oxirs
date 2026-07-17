//! Column-oriented term dictionaries for interned quad storage.
//!
//! [`MemoryStorage`](super::MemoryStorage) stores quads as compact `[u32; 4]`
//! id-tuples instead of five full owned-`Quad` copies. Each quad column
//! (subject, predicate, object, graph name) is interned independently into a
//! [`ColumnDictionary`], mapping the strongly-typed term to a stable `u32` id
//! and back. Interning per column keeps materialization trivial and correct
//! (a stored `Object` is always a valid `Object`, `GraphName::DefaultGraph`
//! needs no special encoding) at the cost of storing a term that appears in
//! several positions once per column — still far cheaper than the previous
//! five owned-`Quad` copies per triple.

use std::collections::HashMap;
use std::hash::Hash;

/// Interns values of a single quad column into stable, monotonically assigned
/// `u32` ids.
///
/// Ids are never reused for a given dictionary instance, so an id resolved from
/// a live index tuple always maps back to the same value. Removing a quad does
/// not reclaim its terms' ids (the terms may still be referenced by other
/// quads, and a per-term reference count would add churn on the hot path);
/// unused ids are reclaimed wholesale when the store is cleared, which replaces
/// the whole `MemoryStorage` (and thus every dictionary) with a fresh one.
#[derive(Debug, Clone)]
pub(crate) struct ColumnDictionary<T: Clone + Eq + Hash> {
    /// id -> value (index into the vector is the id).
    values: Vec<T>,
    /// value -> id.
    ids: HashMap<T, u32>,
}

// Manual `Default` so the column type `T` is not required to be `Default`
// (the derive would add a spurious `T: Default` bound).
impl<T: Clone + Eq + Hash> Default for ColumnDictionary<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone + Eq + Hash> ColumnDictionary<T> {
    /// Create an empty dictionary.
    pub(crate) fn new() -> Self {
        Self {
            values: Vec::new(),
            ids: HashMap::new(),
        }
    }

    /// Return the id for `value`, assigning a fresh id if it is not yet
    /// interned. The value is cloned only on first insertion.
    pub(crate) fn intern(&mut self, value: &T) -> u32 {
        if let Some(&id) = self.ids.get(value) {
            return id;
        }
        let id = self.values.len() as u32;
        self.values.push(value.clone());
        self.ids.insert(value.clone(), id);
        id
    }

    /// Return the id for `value` if it has already been interned.
    pub(crate) fn get_id(&self, value: &T) -> Option<u32> {
        self.ids.get(value).copied()
    }

    /// Resolve an id back to its interned value.
    pub(crate) fn resolve(&self, id: u32) -> Option<&T> {
        self.values.get(id as usize)
    }
}
