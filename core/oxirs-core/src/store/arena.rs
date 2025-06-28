//! Arena-based memory management for efficient RDF data allocation
//!
//! This module provides arena allocators that allocate RDF terms and triples
//! in contiguous memory blocks, reducing memory fragmentation and improving
//! cache locality.

use crate::model::{BlankNode, Literal, NamedNode, Term, Triple, Variable};
use crate::OxirsError;
use bumpalo::Bump;
use crossbeam_utils::thread;
use parking_lot::{Mutex, RwLock};
use std::cell::RefCell;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::mem;
use std::sync::Arc;

thread_local! {
    static THREAD_ARENA: RefCell<Option<Bump>> = RefCell::new(None);
}

/// Arena-allocated string slice with lifetime tied to the arena
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ArenaStr<'arena> {
    value: &'arena str,
}

impl<'arena> ArenaStr<'arena> {
    pub fn as_str(&self) -> &'arena str {
        self.value
    }
}

/// Arena-allocated term reference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArenaTerm<'arena> {
    NamedNode(ArenaStr<'arena>),
    BlankNode(ArenaStr<'arena>),
    Literal {
        value: ArenaStr<'arena>,
        language: Option<ArenaStr<'arena>>,
        datatype: Option<ArenaStr<'arena>>,
    },
    Variable(ArenaStr<'arena>),
}

/// Arena-allocated triple reference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ArenaTriple<'arena> {
    pub subject: ArenaTerm<'arena>,
    pub predicate: ArenaStr<'arena>,
    pub object: ArenaTerm<'arena>,
}

/// Single-threaded arena for allocating RDF data
pub struct LocalArena {
    bump: RefCell<Bump>,
    allocated_bytes: RefCell<usize>,
}

impl LocalArena {
    /// Create a new local arena
    pub fn new() -> Self {
        Self {
            bump: RefCell::new(Bump::new()),
            allocated_bytes: RefCell::new(0),
        }
    }

    /// Create a new arena with a specific capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            bump: RefCell::new(Bump::with_capacity(capacity)),
            allocated_bytes: RefCell::new(0),
        }
    }

    /// Allocate a string in the arena
    pub fn alloc_str(&self, s: &str) -> ArenaStr<'_> {
        let value = unsafe {
            // We use unsafe here because we know the arena will outlive the returned reference
            let bump = &*self.bump.as_ptr();
            bump.alloc_str(s)
        };
        *self.allocated_bytes.borrow_mut() += s.len();
        ArenaStr { value }
    }

    /// Allocate a term in the arena
    pub fn alloc_term<'a>(&'a self, term: &Term) -> ArenaTerm<'a> {
        match term {
            Term::NamedNode(n) => ArenaTerm::NamedNode(self.alloc_str(n.as_str())),
            Term::BlankNode(b) => ArenaTerm::BlankNode(self.alloc_str(b.as_str())),
            Term::Literal(l) => ArenaTerm::Literal {
                value: self.alloc_str(l.value()),
                language: l.language().map(|lang| self.alloc_str(lang)),
                datatype: if l.datatype().as_str() != "http://www.w3.org/2001/XMLSchema#string" {
                    Some(self.alloc_str(l.datatype().as_str()))
                } else {
                    None
                },
            },
            Term::Variable(v) => ArenaTerm::Variable(self.alloc_str(v.as_str())),
            Term::QuotedTriple(_) => panic!("QuotedTriple not supported in arena"),
        }
    }

    /// Allocate a triple in the arena
    pub fn alloc_triple<'a>(&'a self, triple: &Triple) -> ArenaTriple<'a> {
        // Convert subject to term
        let subject_term = match triple.subject() {
            crate::model::Subject::NamedNode(n) => Term::NamedNode(n.clone()),
            crate::model::Subject::BlankNode(b) => Term::BlankNode(b.clone()),
            crate::model::Subject::Variable(v) => Term::Variable(v.clone()),
            crate::model::Subject::QuotedTriple(_) => panic!("QuotedTriple not supported"),
        };

        // Convert object to term
        let object_term = match triple.object() {
            crate::model::Object::NamedNode(n) => Term::NamedNode(n.clone()),
            crate::model::Object::BlankNode(b) => Term::BlankNode(b.clone()),
            crate::model::Object::Literal(l) => Term::Literal(l.clone()),
            crate::model::Object::Variable(v) => Term::Variable(v.clone()),
            crate::model::Object::QuotedTriple(_) => panic!("QuotedTriple not supported"),
        };

        // Get predicate string
        let predicate_str = match triple.predicate() {
            crate::model::Predicate::NamedNode(n) => n.as_str(),
            crate::model::Predicate::Variable(v) => v.as_str(),
        };

        ArenaTriple {
            subject: self.alloc_term(&subject_term),
            predicate: self.alloc_str(predicate_str),
            object: self.alloc_term(&object_term),
        }
    }

    /// Get the total allocated bytes
    pub fn allocated_bytes(&self) -> usize {
        *self.allocated_bytes.borrow()
    }

    /// Reset the arena, freeing all allocations
    pub fn reset(&self) {
        self.bump.borrow_mut().reset();
        *self.allocated_bytes.borrow_mut() = 0;
    }
}

/// Thread-safe arena for concurrent allocation
/// Uses thread-local storage to avoid cross-thread sharing of non-Send types
pub struct ConcurrentArena {
    arena_size: usize,
    total_allocated: Arc<Mutex<usize>>,
}

impl ConcurrentArena {
    /// Create a new concurrent arena with the specified arena size
    pub fn new(arena_size: usize) -> Self {
        Self {
            arena_size,
            total_allocated: Arc::new(Mutex::new(0)),
        }
    }

    /// Allocate a string in the arena using thread-local storage
    pub fn alloc_str(&self, s: &str) -> &'static str {
        let len = s.len();

        THREAD_ARENA.with(|arena_cell| {
            let mut arena_opt = arena_cell.borrow_mut();
            if arena_opt.is_none() {
                *arena_opt = Some(Bump::with_capacity(self.arena_size.max(len * 2)));
            }

            let arena = arena_opt.as_ref().unwrap();
            let allocated = arena.alloc_str(s);
            *self.total_allocated.lock() += len;

            // Unsafe: We're extending the lifetime to 'static
            // This is safe as long as the arena lives as long as the references
            unsafe { std::mem::transmute(allocated) }
        })
    }

    /// Get total allocated bytes across all thread-local arenas
    pub fn total_allocated(&self) -> usize {
        *self.total_allocated.lock()
    }

    /// Get the number of thread-local arenas (simplified to 1 for thread-local impl)
    pub fn arena_count(&self) -> usize {
        THREAD_ARENA.with(
            |arena_cell| {
                if arena_cell.borrow().is_some() {
                    1
                } else {
                    0
                }
            },
        )
    }
}

/// Graph arena that manages memory for an entire RDF graph
pub struct GraphArena<'arena> {
    local_arena: LocalArena,
    term_cache: RefCell<HashMap<Term, ArenaTerm<'arena>>>,
    _phantom: PhantomData<&'arena ()>,
}

impl<'arena> GraphArena<'arena> {
    /// Create a new graph arena
    pub fn new() -> Self {
        Self {
            local_arena: LocalArena::new(),
            term_cache: RefCell::new(HashMap::new()),
            _phantom: PhantomData,
        }
    }

    /// Create a new graph arena with capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            local_arena: LocalArena::with_capacity(capacity),
            term_cache: RefCell::new(HashMap::new()),
            _phantom: PhantomData,
        }
    }

    /// Allocate a term, using cache for deduplication
    pub fn alloc_term(&'arena self, term: &Term) -> ArenaTerm<'arena> {
        let mut cache = self.term_cache.borrow_mut();
        if let Some(&cached) = cache.get(term) {
            return cached;
        }

        let allocated = self.local_arena.alloc_term(term);
        cache.insert(term.clone(), allocated);
        allocated
    }

    /// Allocate a triple
    pub fn alloc_triple(&'arena self, triple: &Triple) -> ArenaTriple<'arena> {
        self.local_arena.alloc_triple(triple)
    }

    /// Get allocated bytes
    pub fn allocated_bytes(&self) -> usize {
        self.local_arena.allocated_bytes()
    }

    /// Get the number of cached terms
    pub fn cached_terms(&self) -> usize {
        self.term_cache.borrow().len()
    }

    /// Clear the arena and cache
    pub fn clear(&self) {
        self.local_arena.reset();
        self.term_cache.borrow_mut().clear();
    }
}

/// Scoped arena for temporary allocations
pub struct ScopedArena<'parent> {
    parent: &'parent LocalArena,
    checkpoint: usize,
}

impl<'parent> ScopedArena<'parent> {
    /// Create a new scoped arena
    pub fn new(parent: &'parent LocalArena) -> Self {
        let checkpoint = parent.allocated_bytes();
        Self { parent, checkpoint }
    }

    /// Allocate a string in the scoped arena
    pub fn alloc_str<'a>(&'a self, s: &str) -> ArenaStr<'a>
    where
        'parent: 'a,
    {
        self.parent.alloc_str(s)
    }

    /// Allocate a term in the scoped arena
    pub fn alloc_term<'a>(&'a self, term: &Term) -> ArenaTerm<'a>
    where
        'parent: 'a,
    {
        self.parent.alloc_term(term)
    }

    /// Get bytes allocated in this scope
    pub fn scope_allocated(&self) -> usize {
        self.parent.allocated_bytes() - self.checkpoint
    }
}

impl<'parent> Drop for ScopedArena<'parent> {
    fn drop(&mut self) {
        // In a real implementation, we could reset to checkpoint
        // For now, we just track the allocation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_local_arena() {
        let arena = LocalArena::new();

        // Test string allocation
        let s1 = arena.alloc_str("hello");
        let s2 = arena.alloc_str("world");
        assert_eq!(s1.as_str(), "hello");
        assert_eq!(s2.as_str(), "world");

        // Test term allocation
        let term = Term::NamedNode(NamedNode::new("http://example.org/test").unwrap());
        let arena_term = arena.alloc_term(&term);
        match arena_term {
            ArenaTerm::NamedNode(s) => assert_eq!(s.as_str(), "http://example.org/test"),
            _ => panic!("Wrong term type"),
        }

        assert!(arena.allocated_bytes() > 0);
    }

    #[test]
    fn test_triple_allocation() {
        let arena = LocalArena::new();

        let triple = Triple::new(
            NamedNode::new("http://s").unwrap(),
            NamedNode::new("http://p").unwrap(),
            Literal::new("object"),
        );

        let arena_triple = arena.alloc_triple(&triple);
        match arena_triple.subject {
            ArenaTerm::NamedNode(s) => assert_eq!(s.as_str(), "http://s"),
            _ => panic!("Wrong subject type"),
        }
        assert_eq!(arena_triple.predicate.as_str(), "http://p");
        match arena_triple.object {
            ArenaTerm::Literal { value, .. } => assert_eq!(value.as_str(), "object"),
            _ => panic!("Wrong object type"),
        }
    }

    #[test]
    fn test_concurrent_arena() {
        let arena = Arc::new(ConcurrentArena::new(1024));

        // Test concurrent allocation
        thread::scope(|s| {
            let handles: Vec<_> = (0..4)
                .map(|i| {
                    let arena_clone = Arc::clone(&arena);
                    s.spawn(move |_| {
                        for j in 0..100 {
                            let string = format!("thread_{}_item_{}", i, j);
                            let allocated = arena_clone.alloc_str(&string);
                            assert_eq!(allocated, string);
                        }
                    })
                })
                .collect();

            for handle in handles {
                handle.join().unwrap();
            }
        })
        .unwrap();

        assert!(arena.total_allocated() > 0);
        assert!(arena.arena_count() >= 1);
    }

    #[test]
    fn test_graph_arena() {
        let arena = GraphArena::new();

        // Test term caching
        let term1 = Term::NamedNode(NamedNode::new("http://example.org/same").unwrap());
        let term2 = term1.clone();

        let allocated1 = arena.alloc_term(&term1);
        let allocated2 = arena.alloc_term(&term2);

        // Should be the same due to caching
        assert_eq!(allocated1, allocated2);
        assert_eq!(arena.cached_terms(), 1);
    }

    #[test]
    fn test_scoped_arena() {
        let parent = LocalArena::new();
        let initial = parent.allocated_bytes();

        {
            let scoped = ScopedArena::new(&parent);
            scoped.alloc_str("temporary");
            assert!(scoped.scope_allocated() > 0);
        }

        // Allocation persists after scope ends (simplified implementation)
        assert!(parent.allocated_bytes() > initial);
    }

    #[test]
    fn test_arena_reset() {
        let arena = LocalArena::new();

        arena.alloc_str("test1");
        arena.alloc_str("test2");
        let allocated = arena.allocated_bytes();
        assert!(allocated > 0);

        arena.reset();
        assert_eq!(arena.allocated_bytes(), 0);
    }
}
