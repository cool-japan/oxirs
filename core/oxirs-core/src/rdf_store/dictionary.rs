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

/// Interns values of a single quad column into stable `u32` ids, reference
/// counting each id so it can be reclaimed the moment its last user goes away.
///
/// Each id carries a reference count. [`intern`](Self::intern) keeps its
/// get-or-create semantics — a repeated value returns the same live id — while
/// callers bracket every *use* of an id with [`retain`](Self::retain) and
/// [`release`](Self::release). When an id's count falls to zero its value is
/// tombstoned (`values[id] = None`), its `ids` entry is dropped, and the id is
/// pushed onto a free list; the next [`intern`](Self::intern) of a *new* value
/// reuses that slot. Reclamation is therefore **synchronous**: between calls,
/// every id present in the `ids` map has a non-zero count, so orphaned ids never
/// accumulate. Ids may be reused, but only after no live tuple references them,
/// so an id resolved from a live index tuple always maps back to the value that
/// tuple recorded.
#[derive(Debug, Clone)]
pub(crate) struct ColumnDictionary<T: Clone + Eq + Hash> {
    /// id -> value (index into the vector is the id). `None` marks a tombstoned
    /// slot whose id currently sits on `free_ids` awaiting reuse.
    values: Vec<Option<T>>,
    /// value -> id (only live, non-tombstoned values are present).
    ids: HashMap<T, u32>,
    /// Parallel to `values`: the reference count of each id (`0` for
    /// tombstoned slots).
    refcounts: Vec<u32>,
    /// Stack of reclaimed ids available for reuse by the next `intern` of a new
    /// value.
    free_ids: Vec<u32>,
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
            refcounts: Vec::new(),
            free_ids: Vec::new(),
        }
    }

    /// Return the id for `value`, assigning an id if it is not yet interned.
    ///
    /// A fresh id is drawn from the free list (reusing a reclaimed slot) when one
    /// is available, otherwise a new slot is appended. The value is cloned only
    /// on first insertion. The id is created with a reference count of zero;
    /// callers take ownership of it with [`retain`](Self::retain).
    pub(crate) fn intern(&mut self, value: &T) -> u32 {
        if let Some(&id) = self.ids.get(value) {
            return id;
        }
        let id = if let Some(reused) = self.free_ids.pop() {
            let idx = reused as usize;
            self.values[idx] = Some(value.clone());
            self.refcounts[idx] = 0;
            reused
        } else {
            let id = self.values.len() as u32;
            self.values.push(Some(value.clone()));
            self.refcounts.push(0);
            id
        };
        self.ids.insert(value.clone(), id);
        id
    }

    /// Return the id for `value` if it has already been interned.
    pub(crate) fn get_id(&self, value: &T) -> Option<u32> {
        self.ids.get(value).copied()
    }

    /// Resolve an id back to its interned value, or `None` if the id is
    /// out of range or currently tombstoned.
    pub(crate) fn resolve(&self, id: u32) -> Option<&T> {
        self.values.get(id as usize).and_then(Option::as_ref)
    }

    /// Take one additional reference on an already-interned id.
    pub(crate) fn retain(&mut self, id: u32) {
        debug_assert!(
            self.resolve(id).is_some(),
            "retain of an id with no live value"
        );
        if let Some(rc) = self.refcounts.get_mut(id as usize) {
            *rc += 1;
        }
    }

    /// Drop one reference on `id`. When the last reference goes away the value is
    /// tombstoned, its `ids` entry removed, and the id pushed onto the free list
    /// for reuse — synchronous reclamation, so a zero-count id is never left
    /// mapped between calls.
    pub(crate) fn release(&mut self, id: u32) {
        let idx = id as usize;
        let rc = match self.refcounts.get_mut(idx) {
            Some(rc) => rc,
            None => return,
        };
        debug_assert!(*rc > 0, "release of an id with no outstanding reference");
        if *rc == 0 {
            return;
        }
        *rc -= 1;
        if *rc == 0 {
            if let Some(value) = self.values[idx].take() {
                self.ids.remove(&value);
            }
            self.free_ids.push(id);
        }
    }
}

impl<T: Clone + Eq + Hash + std::fmt::Display> ColumnDictionary<T> {
    /// Best-effort estimate of this dictionary's resident heap footprint, in
    /// bytes: the `id -> value` vector, the `value -> id` map, the parallel
    /// reference-count vector, and the free-id stack — all at their current
    /// capacities — plus the interned term string bytes (approximated by each
    /// live value's `Display` length). Used by
    /// [`MemoryStorage::size_estimate`](super::storage::MemoryStorage::size_estimate)
    /// for coarse before/after comparisons.
    pub(crate) fn size_estimate(&self) -> usize {
        use std::mem::size_of;
        let structural = self.values.capacity() * size_of::<Option<T>>()
            + self.ids.capacity() * (size_of::<T>() + size_of::<u32>())
            + self.refcounts.capacity() * size_of::<u32>()
            + self.free_ids.capacity() * size_of::<u32>();
        let string_bytes: usize = self
            .values
            .iter()
            .filter_map(Option::as_ref)
            .map(|v| v.to_string().len())
            .sum();
        structural + string_bytes
    }
}

#[cfg(test)]
impl<T: Clone + Eq + Hash> ColumnDictionary<T> {
    /// Number of id slots ever allocated (live plus tombstoned). Bounded across
    /// insert/remove churn thanks to free-list reuse; storage tests assert on
    /// this to prove the dictionary does not grow without bound.
    pub(crate) fn slot_count(&self) -> usize {
        self.values.len()
    }

    /// Number of live (currently-mapped) values.
    pub(crate) fn live_count(&self) -> usize {
        self.ids.len()
    }

    /// Number of reclaimed ids waiting on the free list.
    pub(crate) fn free_count(&self) -> usize {
        self.free_ids.len()
    }
}
