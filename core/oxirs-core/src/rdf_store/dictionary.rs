//! Column-oriented term dictionaries for interned quad storage.
//!
//! [`MemoryStorage`](super::MemoryStorage) stores quads as compact `[u32; 4]`
//! id-tuples instead of five full owned-`Quad` copies. Each quad column
//! (subject, predicate, object, graph name) is interned independently into a
//! [`ColumnDictionary`], mapping the strongly-typed term to a stable `u32` id
//! and back. Interning per column keeps materialization trivial and correct
//! (a stored `Object` is always a valid `Object`, `GraphName::DefaultGraph`
//! needs no special encoding).
//!
//! ## Single-owner interning
//!
//! Each distinct term is materialized **exactly once**, in a reference-counted
//! [`Arc`]. The id vector (`values`) and the reverse-lookup map (`ids`) both hold
//! *the same* `Arc<T>`, so the interned term — including its heap string bytes —
//! is stored a single time and shared by pointer, rather than being cloned into
//! both structures. This halves the per-term footprint of the dictionary (the
//! reverse map previously owned a second full copy of every key), which for the
//! high-cardinality object column is the dominant memory cost of the store.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;

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
///
/// The interned term is held in an [`Arc`] shared between `values` and `ids`, so
/// each distinct term (and its heap string) exists once and is referenced by
/// pointer from both structures.
#[derive(Debug, Clone)]
pub(crate) struct ColumnDictionary<T: Clone + Eq + Hash> {
    /// id -> value (index into the vector is the id). `None` marks a tombstoned
    /// slot whose id currently sits on `free_ids` awaiting reuse. The `Arc`
    /// shares its allocation with the matching `ids` entry.
    values: Vec<Option<Arc<T>>>,
    /// value -> id (only live, non-tombstoned values are present). The key is the
    /// *same* `Arc<T>` stored in `values`, so it adds only a pointer-sized handle
    /// per live term, not a second owned copy of the term.
    ids: HashMap<Arc<T>, u32>,
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
    /// on first insertion — into a single shared [`Arc`] referenced by both
    /// `values` and `ids` (subsequent references are pointer bumps). The id is
    /// created with a reference count of zero; callers take ownership of it with
    /// [`retain`](Self::retain).
    ///
    /// Lookup borrows the `Arc<T>` key as `&T` (`Arc<T>: Borrow<T>`), so probing
    /// never allocates.
    pub(crate) fn intern(&mut self, value: &T) -> u32 {
        if let Some(&id) = self.ids.get(value) {
            return id;
        }
        // First insertion: clone the term once into a shared handle that both the
        // id vector and the lookup map reference by pointer.
        let shared = Arc::new(value.clone());
        let id = if let Some(reused) = self.free_ids.pop() {
            let idx = reused as usize;
            self.values[idx] = Some(Arc::clone(&shared));
            self.refcounts[idx] = 0;
            reused
        } else {
            let id = self.values.len() as u32;
            self.values.push(Some(Arc::clone(&shared)));
            self.refcounts.push(0);
            id
        };
        self.ids.insert(shared, id);
        id
    }

    /// Return the id for `value` if it has already been interned.
    pub(crate) fn get_id(&self, value: &T) -> Option<u32> {
        self.ids.get(value).copied()
    }

    /// Resolve an id back to its interned value, or `None` if the id is
    /// out of range or currently tombstoned.
    pub(crate) fn resolve(&self, id: u32) -> Option<&T> {
        self.values
            .get(id as usize)
            .and_then(|slot| slot.as_deref())
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

    /// Release excess reserved capacity in every backing allocation back to the
    /// allocator. A bulk load grows the `Vec`/`HashMap` by repeated doubling and
    /// leaves them over-provisioned by up to ~2x; calling this once the load is
    /// complete returns that slack. Preserves all live ids and values (it only
    /// changes capacities, never contents), so ids resolved before and after a
    /// shrink are identical.
    pub(crate) fn shrink_to_fit(&mut self) {
        self.values.shrink_to_fit();
        self.ids.shrink_to_fit();
        self.refcounts.shrink_to_fit();
        self.free_ids.shrink_to_fit();
    }

    /// Reserve room for `additional` more distinct values across the id vector,
    /// lookup map, and refcount vector, so a bulk load can size its allocations
    /// up front instead of repeatedly doubling them (which transiently holds both
    /// the old and new backing buffers during each rehash). The free list is left
    /// unreserved — it only grows on deletion.
    pub(crate) fn reserve(&mut self, additional: usize) {
        self.values.reserve(additional);
        self.ids.reserve(additional);
        self.refcounts.reserve(additional);
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
            if let Some(shared) = self.values[idx].take() {
                // Drop the lookup entry keyed by the same term; the shared
                // allocation frees once this local handle also drops at block end.
                self.ids.remove(shared.as_ref());
            }
            self.free_ids.push(id);
        }
    }
}

impl<T: Clone + Eq + Hash + std::fmt::Display> ColumnDictionary<T> {
    /// Best-effort estimate of this dictionary's resident heap footprint, in
    /// bytes: the `id -> value` vector (pointer-sized `Arc` handles), the
    /// `value -> id` map (`Arc` handle + `u32` per bucket), the parallel
    /// reference-count vector, and the free-id stack — all at their current
    /// capacities — plus, for each *distinct* live term, the shared `Arc`
    /// allocation it is stored in exactly once (strong/weak counts, the inline
    /// `T`, and its `Display`-approximated heap string bytes). Because each term
    /// is now single-owned, this is a much tighter estimate than the previous
    /// design, whose reverse map silently duplicated every key. Used by
    /// [`MemoryStorage::size_estimate`](super::storage::MemoryStorage::size_estimate)
    /// for coarse before/after comparisons.
    pub(crate) fn size_estimate(&self) -> usize {
        use std::mem::size_of;
        let structural = self.values.capacity() * size_of::<Option<Arc<T>>>()
            + self.ids.capacity() * (size_of::<Arc<T>>() + size_of::<u32>())
            + self.refcounts.capacity() * size_of::<u32>()
            + self.free_ids.capacity() * size_of::<u32>();
        // Each live term is materialized once in a shared allocation: the Arc's
        // strong and weak counts, the inline `T`, and its heap string bytes.
        let per_term_header = 2 * size_of::<usize>() + size_of::<T>();
        let mut live_terms = 0usize;
        let string_bytes: usize = self
            .values
            .iter()
            .filter_map(Option::as_ref)
            .map(|v| {
                live_terms += 1;
                v.to_string().len()
            })
            .sum();
        structural + live_terms * per_term_header + string_bytes
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A hand-rolled column type whose `Hash` deliberately collapses every value
    /// into the *same* hash bucket, so the reverse-lookup map must fall back on
    /// `Eq` to tell distinct terms apart. Interning, resolving, id reuse, and
    /// removal must all stay correct even when every key collides — the scenario
    /// a real hash function makes vanishingly rare but never impossible.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct Collider(String);

    impl Hash for Collider {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            // Constant hash: force every Collider into one bucket.
            0u8.hash(state);
        }
    }

    impl std::fmt::Display for Collider {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    fn c(s: &str) -> Collider {
        Collider(s.to_string())
    }

    /// Distinct-but-colliding values each get their own id, resolve back to the
    /// value that was interned, and repeated interning is idempotent.
    #[test]
    fn colliding_values_get_distinct_ids_and_resolve_correctly() {
        let mut dict: ColumnDictionary<Collider> = ColumnDictionary::new();
        let a = dict.intern(&c("a"));
        let b = dict.intern(&c("b"));
        let d = dict.intern(&c("c"));
        assert_ne!(a, b);
        assert_ne!(b, d);
        assert_ne!(a, d);
        // Re-interning returns the same id (get-or-create), no new slot.
        assert_eq!(dict.intern(&c("a")), a);
        assert_eq!(dict.intern(&c("b")), b);
        assert_eq!(dict.slot_count(), 3);
        assert_eq!(dict.live_count(), 3);
        assert_eq!(dict.resolve(a), Some(&c("a")));
        assert_eq!(dict.resolve(b), Some(&c("b")));
        assert_eq!(dict.resolve(d), Some(&c("c")));
        assert_eq!(dict.get_id(&c("b")), Some(b));
        assert_eq!(dict.get_id(&c("absent")), None);
    }

    /// With every value colliding, releasing one term's last reference must
    /// tombstone exactly that term, free its id for reuse, and leave the other
    /// colliding terms untouched and still resolvable.
    #[test]
    fn release_under_full_collision_reclaims_only_the_right_id() {
        let mut dict: ColumnDictionary<Collider> = ColumnDictionary::new();
        let a = dict.intern(&c("a"));
        let b = dict.intern(&c("b"));
        let d = dict.intern(&c("c"));
        for id in [a, b, d] {
            dict.retain(id);
        }
        assert_eq!(dict.live_count(), 3);

        // Release b's only reference: b is reclaimed, a and c survive.
        dict.release(b);
        assert_eq!(dict.get_id(&c("b")), None);
        assert_eq!(dict.resolve(b), None);
        assert_eq!(dict.live_count(), 2);
        assert_eq!(dict.free_count(), 1);
        assert_eq!(dict.resolve(a), Some(&c("a")));
        assert_eq!(dict.resolve(d), Some(&c("c")));
        assert_eq!(dict.get_id(&c("a")), Some(a));
        assert_eq!(dict.get_id(&c("c")), Some(d));

        // A new colliding term reuses b's freed slot.
        let e = dict.intern(&c("e"));
        assert_eq!(e, b, "freed id must be reused");
        assert_eq!(dict.slot_count(), 3, "no new slot allocated");
        assert_eq!(dict.resolve(e), Some(&c("e")));
        assert_eq!(dict.resolve(a), Some(&c("a")));
    }

    /// Reference counting is honored: a term interned/retained twice stays live
    /// after one release and is reclaimed only when the last reference drops.
    #[test]
    fn refcount_keeps_shared_term_until_last_release() {
        let mut dict: ColumnDictionary<Collider> = ColumnDictionary::new();
        let a = dict.intern(&c("a"));
        dict.retain(a);
        dict.retain(a); // two references on the same id
        assert_eq!(dict.live_count(), 1);

        dict.release(a);
        assert_eq!(dict.get_id(&c("a")), Some(a), "one reference remains");
        assert_eq!(dict.resolve(a), Some(&c("a")));

        dict.release(a);
        assert_eq!(dict.get_id(&c("a")), None, "last reference gone");
        assert_eq!(dict.resolve(a), None);
        assert_eq!(dict.free_count(), 1);
    }

    /// After a shrink the dictionary must resolve every live id to exactly the
    /// value it recorded, and reserve+shrink must not disturb contents.
    #[test]
    fn shrink_and_reserve_preserve_all_live_mappings() {
        let mut dict: ColumnDictionary<Collider> = ColumnDictionary::new();
        dict.reserve(64);
        let mut ids = Vec::new();
        for i in 0..50 {
            let id = dict.intern(&c(&format!("v{i}")));
            dict.retain(id);
            ids.push((i, id));
        }
        // Release a scattered third of them to leave tombstones/free ids.
        for &(_, id) in ids.iter().filter(|(i, _)| i % 3 == 0) {
            dict.release(id);
        }
        dict.shrink_to_fit();
        // Every still-live term resolves to its own value; released ones are gone.
        for &(i, id) in &ids {
            if i % 3 == 0 {
                assert_eq!(dict.resolve(id), None, "released v{i} must be gone");
            } else {
                assert_eq!(
                    dict.resolve(id),
                    Some(&c(&format!("v{i}"))),
                    "live v{i} must survive shrink"
                );
            }
        }
    }
}
