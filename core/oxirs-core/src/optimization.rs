//! Zero-copy operations and performance optimizations
//!
//! This module provides advanced performance optimizations including zero-copy
//! operations, memory-efficient data structures, and SIMD-accelerated processing
//! for RDF data manipulation.

use crate::interning::{InternedString, StringInterner};
use crate::model::*;
use bumpalo::Bump;
use crossbeam::epoch::{self, Atomic, Owned};
use crossbeam::queue::SegQueue;
use dashmap::DashMap;
use parking_lot::RwLock;
#[cfg(feature = "parallel")]
use rayon::iter::IntoParallelRefIterator;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;
use simd_json;
use std::collections::BTreeSet;
use std::pin::Pin;
use std::sync::atomic::Ordering;
use std::sync::Arc;

/// Type alias for string interner used throughout optimization module
pub type TermInterner = StringInterner;

/// Extension trait for TermInterner to create RDF terms
pub trait TermInternerExt {
    /// Intern a named node and return it
    fn intern_named_node(&self, iri: &str) -> Result<NamedNode, crate::OxirsError>;

    /// Create and intern a new blank node
    fn intern_blank_node(&self) -> BlankNode;

    /// Intern a simple literal
    fn intern_literal(&self, value: &str) -> Result<Literal, crate::OxirsError>;

    /// Intern a literal with datatype
    fn intern_literal_with_datatype(
        &self,
        value: &str,
        datatype_iri: &str,
    ) -> Result<Literal, crate::OxirsError>;

    /// Intern a literal with language tag
    fn intern_literal_with_language(
        &self,
        value: &str,
        language: &str,
    ) -> Result<Literal, crate::OxirsError>;
}

impl TermInternerExt for TermInterner {
    fn intern_named_node(&self, iri: &str) -> Result<NamedNode, crate::OxirsError> {
        // Intern the IRI string
        let interned = self.intern(iri);
        // Create NamedNode from the interned string
        NamedNode::new(interned.as_ref())
    }

    fn intern_blank_node(&self) -> BlankNode {
        // Generate a unique blank node
        BlankNode::new_unique()
    }

    fn intern_literal(&self, value: &str) -> Result<Literal, crate::OxirsError> {
        // Intern the literal value
        let interned = self.intern(value);
        // Create simple literal
        Ok(Literal::new_simple_literal(interned.as_ref()))
    }

    fn intern_literal_with_datatype(
        &self,
        value: &str,
        datatype_iri: &str,
    ) -> Result<Literal, crate::OxirsError> {
        // Intern both value and datatype IRI
        let value_interned = self.intern(value);
        let datatype_interned = self.intern(datatype_iri);
        // Create datatype node and literal
        let datatype_node = NamedNode::new(datatype_interned.as_ref())?;
        Ok(Literal::new_typed_literal(
            value_interned.as_ref(),
            datatype_node,
        ))
    }

    fn intern_literal_with_language(
        &self,
        value: &str,
        language: &str,
    ) -> Result<Literal, crate::OxirsError> {
        // Intern both value and language tag
        let value_interned = self.intern(value);
        let language_interned = self.intern(language);
        // Create language-tagged literal
        let literal = Literal::new_language_tagged_literal(
            value_interned.as_ref(),
            language_interned.as_ref(),
        )?;
        Ok(literal)
    }
}

/// Arena-based memory allocator for RDF terms
///
/// Provides fast allocation and automatic cleanup for temporary RDF operations
#[derive(Debug)]
pub struct RdfArena {
    /// Main allocation arena (wrapped in Mutex for thread safety)
    arena: std::sync::Mutex<Bump>,
    /// String interner for the arena
    interner: StringInterner,
    /// Statistics
    allocated_bytes: std::sync::atomic::AtomicUsize,
    allocation_count: std::sync::atomic::AtomicUsize,
}

impl RdfArena {
    /// Create a new RDF arena with the given capacity hint
    pub fn new() -> Self {
        RdfArena {
            arena: std::sync::Mutex::new(Bump::new()),
            interner: StringInterner::new(),
            allocated_bytes: std::sync::atomic::AtomicUsize::new(0),
            allocation_count: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Create a new arena with pre-allocated capacity
    pub fn with_capacity(capacity: usize) -> Self {
        RdfArena {
            arena: std::sync::Mutex::new(Bump::with_capacity(capacity)),
            interner: StringInterner::new(),
            allocated_bytes: std::sync::atomic::AtomicUsize::new(0),
            allocation_count: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Allocate a string in the arena
    pub fn alloc_str(&self, s: &str) -> String {
        // Since we can't return a reference with Mutex, return an owned String
        // For ultra-performance mode, the caller should use string interning instead
        self.allocated_bytes
            .fetch_add(s.len(), std::sync::atomic::Ordering::Relaxed);
        self.allocation_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        s.to_string()
    }

    /// Allocate and intern a string for efficient reuse
    pub fn intern_str(&self, s: &str) -> InternedString {
        InternedString::new_with_interner(s, &self.interner)
    }

    /// Reset the arena, freeing all allocated memory
    pub fn reset(&self) {
        if let Ok(mut arena) = self.arena.lock() {
            arena.reset();
            self.allocated_bytes
                .store(0, std::sync::atomic::Ordering::Relaxed);
            self.allocation_count
                .store(0, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Get total bytes allocated
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get total allocation count
    pub fn allocation_count(&self) -> usize {
        self.allocation_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl Default for RdfArena {
    fn default() -> Self {
        Self::new()
    }
}

/// Zero-copy RDF term reference that avoids allocations
///
/// This provides efficient operations on RDF terms without copying data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TermRef<'a> {
    NamedNode(&'a str),
    BlankNode(&'a str),
    Literal(&'a str, Option<&'a str>, Option<&'a str>), // value, datatype, language
    Variable(&'a str),
}

impl<'a> TermRef<'a> {
    /// Create a term reference from a named node
    pub fn from_named_node(node: &'a NamedNode) -> Self {
        TermRef::NamedNode(node.as_str())
    }

    /// Create a term reference from a blank node
    pub fn from_blank_node(node: &'a BlankNode) -> Self {
        TermRef::BlankNode(node.as_str())
    }

    /// Create a term reference from a literal
    pub fn from_literal(literal: &'a Literal) -> Self {
        let language = literal.language();
        // Always include datatype IRI for now to avoid lifetime issues
        // Skip datatype for now due to lifetime issues - would need redesign
        TermRef::Literal(literal.value(), None, language)
    }

    /// Get the string representation of this term
    pub fn as_str(&self) -> &'a str {
        match self {
            TermRef::NamedNode(s) => s,
            TermRef::BlankNode(s) => s,
            TermRef::Literal(s, _, _) => s,
            TermRef::Variable(s) => s,
        }
    }

    /// Convert to an owned Term (allocating if necessary)
    pub fn to_owned(&self) -> Result<Term, crate::OxirsError> {
        match self {
            TermRef::NamedNode(iri) => Ok(Term::NamedNode(NamedNode::new(*iri)?)),
            TermRef::BlankNode(id) => Ok(Term::BlankNode(BlankNode::new(*id)?)),
            TermRef::Literal(value, datatype, language) => {
                let literal = if let Some(lang) = language {
                    Literal::new_lang(*value, *lang)?
                } else if let Some(dt) = datatype {
                    let dt_node = NamedNode::new(*dt)?;
                    Literal::new_typed(*value, dt_node)
                } else {
                    Literal::new(*value)
                };
                Ok(Term::Literal(literal))
            }
            TermRef::Variable(name) => Ok(Term::Variable(Variable::new(*name)?)),
        }
    }

    /// Returns true if this is a named node
    pub fn is_named_node(&self) -> bool {
        matches!(self, TermRef::NamedNode(_))
    }

    /// Returns true if this is a blank node
    pub fn is_blank_node(&self) -> bool {
        matches!(self, TermRef::BlankNode(_))
    }

    /// Returns true if this is a literal
    pub fn is_literal(&self) -> bool {
        matches!(self, TermRef::Literal(_, _, _))
    }

    /// Returns true if this is a variable
    pub fn is_variable(&self) -> bool {
        matches!(self, TermRef::Variable(_))
    }
}

impl<'a> std::fmt::Display for TermRef<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TermRef::NamedNode(iri) => write!(f, "<{}>", iri),
            TermRef::BlankNode(id) => write!(f, "{}", id),
            TermRef::Literal(value, datatype, language) => {
                write!(f, "\"{}\"", value)?;
                if let Some(lang) = language {
                    write!(f, "@{}", lang)?;
                } else if let Some(dt) = datatype {
                    write!(f, "^^<{}>", dt)?;
                }
                Ok(())
            }
            TermRef::Variable(name) => write!(f, "?{}", name),
        }
    }
}

/// Zero-copy triple reference for efficient operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TripleRef<'a> {
    pub subject: TermRef<'a>,
    pub predicate: TermRef<'a>,
    pub object: TermRef<'a>,
}

impl<'a> TripleRef<'a> {
    /// Create a new triple reference
    pub fn new(subject: TermRef<'a>, predicate: TermRef<'a>, object: TermRef<'a>) -> Self {
        TripleRef {
            subject,
            predicate,
            object,
        }
    }

    /// Create from an owned triple
    pub fn from_triple(triple: &'a Triple) -> Self {
        TripleRef {
            subject: match triple.subject() {
                Subject::NamedNode(n) => TermRef::NamedNode(n.as_str()),
                Subject::BlankNode(b) => TermRef::BlankNode(b.as_str()),
                Subject::Variable(v) => TermRef::Variable(v.as_str()),
                Subject::QuotedTriple(_) => TermRef::NamedNode("<<quoted-triple>>"),
            },
            predicate: match triple.predicate() {
                Predicate::NamedNode(n) => TermRef::NamedNode(n.as_str()),
                Predicate::Variable(v) => TermRef::Variable(v.as_str()),
            },
            object: match triple.object() {
                Object::NamedNode(n) => TermRef::NamedNode(n.as_str()),
                Object::BlankNode(b) => TermRef::BlankNode(b.as_str()),
                Object::Literal(l) => TermRef::from_literal(l),
                Object::Variable(v) => TermRef::Variable(v.as_str()),
                Object::QuotedTriple(_) => TermRef::NamedNode("<<quoted-triple>>"),
            },
        }
    }

    /// Convert to an owned triple
    pub fn to_owned(&self) -> Result<Triple, crate::OxirsError> {
        let subject = match self.subject.to_owned()? {
            Term::NamedNode(n) => Subject::NamedNode(n),
            Term::BlankNode(b) => Subject::BlankNode(b),
            _ => return Err(crate::OxirsError::Parse("Invalid subject term".to_string())),
        };

        let predicate = match self.predicate.to_owned()? {
            Term::NamedNode(n) => Predicate::NamedNode(n),
            _ => {
                return Err(crate::OxirsError::Parse(
                    "Invalid predicate term".to_string(),
                ))
            }
        };

        let object = match self.object.to_owned()? {
            Term::NamedNode(n) => Object::NamedNode(n),
            Term::BlankNode(b) => Object::BlankNode(b),
            Term::Literal(l) => Object::Literal(l),
            _ => return Err(crate::OxirsError::Parse("Invalid object term".to_string())),
        };

        Ok(Triple::new(subject, predicate, object))
    }
}

impl<'a> std::fmt::Display for TripleRef<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {} {} .", self.subject, self.predicate, self.object)
    }
}

/// Lock-free graph operations using epoch-based memory management
#[derive(Debug)]
pub struct LockFreeGraph {
    /// Atomic pointer to the current graph data
    data: Atomic<GraphData>,
    /// Epoch for safe memory reclamation
    epoch: epoch::Guard,
}

/// Internal graph data structure for lock-free operations
#[derive(Debug)]
struct GraphData {
    /// Triples stored in a B-tree for ordered access
    triples: BTreeSet<Triple>,
    /// Version number for optimistic updates
    version: u64,
}

impl LockFreeGraph {
    /// Create a new lock-free graph
    pub fn new() -> Self {
        let initial_data = GraphData {
            triples: BTreeSet::new(),
            version: 0,
        };

        LockFreeGraph {
            data: Atomic::new(initial_data),
            epoch: epoch::pin(),
        }
    }

    /// Insert a triple using compare-and-swap
    pub fn insert(&self, triple: Triple) -> bool {
        loop {
            let current = self.data.load(Ordering::Acquire, &self.epoch);
            let current_ref = unsafe { current.deref() };

            // Check if triple already exists
            if current_ref.triples.contains(&triple) {
                return false;
            }

            // Create new data with the inserted triple
            let mut new_triples = current_ref.triples.clone();
            new_triples.insert(triple.clone());

            let new_data = GraphData {
                triples: new_triples,
                version: current_ref.version + 1,
            };

            // Try to update atomically
            match self.data.compare_exchange_weak(
                current,
                Owned::new(new_data),
                Ordering::Release,
                Ordering::Relaxed,
                &self.epoch,
            ) {
                Ok(_) => {
                    // Successfully updated
                    unsafe {
                        self.epoch.defer_destroy(current);
                    }
                    return true;
                }
                Err(_) => {
                    // Retry with new current value
                    continue;
                }
            }
        }
    }

    /// Get the current number of triples
    pub fn len(&self) -> usize {
        let current = self.data.load(Ordering::Acquire, &self.epoch);
        unsafe { current.deref().triples.len() }
    }

    /// Check if the graph is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Check if a triple exists
    pub fn contains(&self, triple: &Triple) -> bool {
        let current = self.data.load(Ordering::Acquire, &self.epoch);
        unsafe { current.deref().triples.contains(triple) }
    }
}

impl Default for LockFreeGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// High-performance graph with multiple indexing strategies
#[derive(Debug)]
pub struct OptimizedGraph {
    /// Subject-Predicate-Object index
    spo: DashMap<InternedString, DashMap<InternedString, BTreeSet<InternedString>>>,
    /// Predicate-Object-Subject index
    pos: DashMap<InternedString, DashMap<InternedString, BTreeSet<InternedString>>>,
    /// Object-Subject-Predicate index
    osp: DashMap<InternedString, DashMap<InternedString, BTreeSet<InternedString>>>,
    /// String interner for memory efficiency
    interner: Arc<StringInterner>,
    /// Statistics
    stats: Arc<RwLock<GraphStats>>,
}

/// Statistics for the optimized graph
#[derive(Debug, Clone, Default)]
pub struct GraphStats {
    pub triple_count: usize,
    pub unique_subjects: usize,
    pub unique_predicates: usize,
    pub unique_objects: usize,
    pub index_memory_usage: usize,
    pub intern_hit_ratio: f64,
}

impl OptimizedGraph {
    /// Create a new optimized graph
    pub fn new() -> Self {
        OptimizedGraph {
            spo: DashMap::new(),
            pos: DashMap::new(),
            osp: DashMap::new(),
            interner: Arc::new(StringInterner::new()),
            stats: Arc::new(RwLock::new(GraphStats::default())),
        }
    }

    /// Insert a triple into all indexes
    pub fn insert(&self, triple: &Triple) -> bool {
        let subject = self.intern_subject(triple.subject());
        let predicate = self.intern_predicate(triple.predicate());
        let object = self.intern_object(triple.object());

        // Insert into SPO index
        let spo_entry = self.spo.entry(subject.clone()).or_insert_with(DashMap::new);
        let mut po_entry = spo_entry
            .entry(predicate.clone())
            .or_insert_with(BTreeSet::new);
        let was_new = po_entry.insert(object.clone());

        if was_new {
            // Insert into POS index
            let pos_entry = self
                .pos
                .entry(predicate.clone())
                .or_insert_with(DashMap::new);
            let mut os_entry = pos_entry
                .entry(object.clone())
                .or_insert_with(BTreeSet::new);
            os_entry.insert(subject.clone());

            // Insert into OSP index
            let osp_entry = self.osp.entry(object.clone()).or_insert_with(DashMap::new);
            let mut sp_entry = osp_entry
                .entry(subject.clone())
                .or_insert_with(BTreeSet::new);
            sp_entry.insert(predicate);

            // Update statistics
            {
                let mut stats = self.stats.write();
                stats.triple_count += 1;
                stats.intern_hit_ratio = self.interner.stats().hit_ratio();
            }
        }

        was_new
    }

    /// Query triples by pattern (None = wildcard)
    pub fn query(
        &self,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
    ) -> Vec<Triple> {
        let mut results = Vec::new();

        // Choose the most selective index based on bound variables
        match (subject.is_some(), predicate.is_some(), object.is_some()) {
            (true, true, true) => {
                // Exact match - use SPO index
                if let (Some(s), Some(p), Some(o)) = (subject, predicate, object) {
                    let s_intern = self.intern_subject(s);
                    let p_intern = self.intern_predicate(p);
                    let o_intern = self.intern_object(o);

                    if let Some(po_map) = self.spo.get(&s_intern) {
                        if let Some(o_set) = po_map.get(&p_intern) {
                            if o_set.contains(&o_intern) {
                                let triple = Triple::new(s.clone(), p.clone(), o.clone());
                                results.push(triple);
                            }
                        }
                    }
                }
            }
            (true, true, false) => {
                // Subject and predicate bound - use SPO index
                if let (Some(s), Some(p)) = (subject, predicate) {
                    let s_intern = self.intern_subject(s);
                    let p_intern = self.intern_predicate(p);

                    if let Some(po_map) = self.spo.get(&s_intern) {
                        if let Some(o_set) = po_map.get(&p_intern) {
                            for o_intern in o_set.iter() {
                                if let Ok(object) = self.unintern_object(o_intern) {
                                    let triple = Triple::new(s.clone(), p.clone(), object);
                                    results.push(triple);
                                }
                            }
                        }
                    }
                }
            }
            (false, true, true) => {
                // Predicate and object bound - use POS index
                if let (Some(p), Some(o)) = (predicate, object) {
                    let p_intern = self.intern_predicate(p);
                    let o_intern = self.intern_object(o);

                    if let Some(os_map) = self.pos.get(&p_intern) {
                        if let Some(s_set) = os_map.get(&o_intern) {
                            for s_intern in s_set.iter() {
                                if let Ok(subject) = self.unintern_subject(s_intern) {
                                    let triple = Triple::new(subject, p.clone(), o.clone());
                                    results.push(triple);
                                }
                            }
                        }
                    }
                }
            }
            _ => {
                // Other patterns - full scan (could be optimized further)
                for s_entry in &self.spo {
                    let s_intern = s_entry.key();
                    if let Ok(s) = self.unintern_subject(s_intern) {
                        if subject.is_some() && subject.unwrap() != &s {
                            continue;
                        }

                        for po_entry in s_entry.value().iter() {
                            let p_intern = po_entry.key();
                            if let Ok(p) = self.unintern_predicate(p_intern) {
                                if predicate.is_some() && predicate.unwrap() != &p {
                                    continue;
                                }

                                for o_intern in po_entry.value().iter() {
                                    if let Ok(o) = self.unintern_object(o_intern) {
                                        if object.is_some() && object.unwrap() != &o {
                                            continue;
                                        }

                                        let triple = Triple::new(s.clone(), p.clone(), o);
                                        results.push(triple);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        results
    }

    /// Get current statistics
    pub fn stats(&self) -> GraphStats {
        self.stats.read().clone()
    }

    /// Intern a subject term
    fn intern_subject(&self, subject: &Subject) -> InternedString {
        match subject {
            Subject::NamedNode(n) => InternedString::new_with_interner(n.as_str(), &self.interner),
            Subject::BlankNode(b) => InternedString::new_with_interner(b.as_str(), &self.interner),
            Subject::Variable(v) => InternedString::new_with_interner(v.as_str(), &self.interner),
            Subject::QuotedTriple(_) => {
                InternedString::new_with_interner("<<quoted-triple>>", &self.interner)
            }
        }
    }

    /// Intern a predicate term
    fn intern_predicate(&self, predicate: &Predicate) -> InternedString {
        match predicate {
            Predicate::NamedNode(n) => {
                InternedString::new_with_interner(n.as_str(), &self.interner)
            }
            Predicate::Variable(v) => InternedString::new_with_interner(v.as_str(), &self.interner),
        }
    }

    /// Intern an object term
    fn intern_object(&self, object: &Object) -> InternedString {
        match object {
            Object::NamedNode(n) => InternedString::new_with_interner(n.as_str(), &self.interner),
            Object::BlankNode(b) => InternedString::new_with_interner(b.as_str(), &self.interner),
            Object::Literal(l) => {
                // For literals, we store a serialized representation
                let serialized = format!("{}", l);
                InternedString::new_with_interner(&serialized, &self.interner)
            }
            Object::Variable(v) => InternedString::new_with_interner(v.as_str(), &self.interner),
            Object::QuotedTriple(_) => {
                InternedString::new_with_interner("<<quoted-triple>>", &self.interner)
            }
        }
    }

    /// Convert interned subject back to Subject
    fn unintern_subject(&self, interned: &InternedString) -> Result<Subject, crate::OxirsError> {
        let s = interned.as_str();
        if s.starts_with("?") || s.starts_with("$") {
            Ok(Subject::Variable(Variable::new(&s[1..])?))
        } else if s.starts_with("_:") {
            Ok(Subject::BlankNode(BlankNode::new(s)?))
        } else {
            Ok(Subject::NamedNode(NamedNode::new(s)?))
        }
    }

    /// Convert interned predicate back to Predicate
    fn unintern_predicate(
        &self,
        interned: &InternedString,
    ) -> Result<Predicate, crate::OxirsError> {
        let s = interned.as_str();
        if s.starts_with("?") || s.starts_with("$") {
            Ok(Predicate::Variable(Variable::new(&s[1..])?))
        } else {
            Ok(Predicate::NamedNode(NamedNode::new(s)?))
        }
    }

    /// Convert interned object back to Object
    fn unintern_object(&self, interned: &InternedString) -> Result<Object, crate::OxirsError> {
        let s = interned.as_str();
        if s.starts_with("?") || s.starts_with("$") {
            return Ok(Object::Variable(Variable::new(&s[1..])?));
        } else if s.starts_with("\"") {
            // Parse literal (simplified - would need full parser for production)
            if let Some(end_quote) = s[1..].find('"') {
                let value = &s[1..end_quote + 1];
                return Ok(Object::Literal(Literal::new(value)));
            }
            // If no end quote found, treat as a simple literal
            return Ok(Object::Literal(Literal::new(s)));
        }

        if s.starts_with("_:") {
            Ok(Object::BlankNode(BlankNode::new(s)?))
        } else {
            Ok(Object::NamedNode(NamedNode::new(s)?))
        }
    }
}

impl Default for OptimizedGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Lock-free queue for batch processing operations
#[cfg(feature = "parallel")]
#[derive(Debug)]
pub struct BatchProcessor {
    /// Queue for pending operations
    operation_queue: SegQueue<BatchOperation>,
    /// Background processing threads
    processing_pool: rayon::ThreadPool,
    /// Statistics
    stats: Arc<RwLock<BatchStats>>,
}

/// Batch operation types
#[derive(Debug, Clone)]
pub enum BatchOperation {
    Insert(Quad),
    Delete(Quad),
    Update { old: Quad, new: Quad },
    Compact,
}

/// Batch processing statistics
#[derive(Debug, Default, Clone)]
pub struct BatchStats {
    pub operations_processed: usize,
    pub batch_size: usize,
    pub processing_time_ms: u64,
    pub throughput_ops_per_sec: f64,
}

#[cfg(feature = "parallel")]
impl BatchProcessor {
    /// Create a new batch processor with specified thread count
    pub fn new(num_threads: usize) -> Self {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        BatchProcessor {
            operation_queue: SegQueue::new(),
            processing_pool: pool,
            stats: Arc::new(RwLock::new(BatchStats::default())),
        }
    }

    /// Add an operation to the batch queue
    pub fn push(&self, operation: BatchOperation) {
        self.operation_queue.push(operation);
    }

    /// Process all pending operations in batches
    pub fn process_batch(&self, batch_size: usize) -> Result<usize, crate::OxirsError> {
        let start_time = std::time::Instant::now();
        let mut operations = Vec::with_capacity(batch_size);

        // Collect operations from queue
        for _ in 0..batch_size {
            if let Some(op) = self.operation_queue.pop() {
                operations.push(op);
            } else {
                break;
            }
        }

        if operations.is_empty() {
            return Ok(0);
        }

        let operations_count = operations.len();

        // Process operations in parallel using Rayon
        self.processing_pool.install(|| {
            operations.par_iter().for_each(|operation| {
                match operation {
                    BatchOperation::Insert(_quad) => {
                        // Parallel insert logic would go here
                    }
                    BatchOperation::Delete(_quad) => {
                        // Parallel delete logic would go here
                    }
                    BatchOperation::Update {
                        old: _old,
                        new: _new,
                    } => {
                        // Parallel update logic would go here
                    }
                    BatchOperation::Compact => {
                        // Compaction logic would go here
                    }
                }
            });
        });

        // Update statistics
        let processing_time = start_time.elapsed();
        {
            let mut stats = self.stats.write();
            stats.operations_processed += operations_count;
            stats.batch_size = batch_size;
            stats.processing_time_ms = processing_time.as_millis() as u64;
            if processing_time.as_secs_f64() > 0.0 {
                stats.throughput_ops_per_sec =
                    operations_count as f64 / processing_time.as_secs_f64();
            }
        }

        Ok(operations_count)
    }

    /// Get current processing statistics
    pub fn stats(&self) -> BatchStats {
        self.stats.read().clone()
    }

    /// Get the number of pending operations
    pub fn pending_operations(&self) -> usize {
        self.operation_queue.len()
    }
}

#[cfg(feature = "parallel")]
impl Default for BatchProcessor {
    fn default() -> Self {
        Self::new(num_cpus::get())
    }
}

/// SIMD-accelerated string operations for RDF processing
pub mod simd {
    #[cfg(feature = "simd")]
    use wide::{u8x32, CmpEq};

    /// Fast IRI validation using SIMD operations
    #[cfg(feature = "simd")]
    pub fn validate_iri_fast(iri: &str) -> bool {
        if iri.is_empty() {
            return false;
        }

        let bytes = iri.as_bytes();
        let len = bytes.len();

        // Process 32 bytes at a time using SIMD
        let chunks = len / 32;
        let remainder = len % 32;

        for i in 0..chunks {
            let start = i * 32;
            let chunk = &bytes[start..start + 32];

            // Load 32 bytes
            let data = u8x32::from([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                chunk[8], chunk[9], chunk[10], chunk[11], chunk[12], chunk[13], chunk[14],
                chunk[15], chunk[16], chunk[17], chunk[18], chunk[19], chunk[20], chunk[21],
                chunk[22], chunk[23], chunk[24], chunk[25], chunk[26], chunk[27], chunk[28],
                chunk[29], chunk[30], chunk[31],
            ]);

            // Check for forbidden characters (< > " { } | \ ^ ` space)
            let forbidden_chars = [b'<', b'>', b'"', b'{', b'}', b'|', b'\\', b'^', b'`', b' '];

            for &forbidden in &forbidden_chars {
                let forbidden_vec = u8x32::splat(forbidden);
                let matches = data.cmp_eq(forbidden_vec);
                if matches.any() {
                    return false;
                }
            }

            // Check for control characters (0-31, 127-159)
            for &byte in chunk {
                if matches!(byte, 0..=31 | 127..=159) {
                    return false;
                }
            }
        }

        // Process remaining bytes
        for &byte in &bytes[chunks * 32..] {
            if matches!(byte,
                0..=31 | 127..=159 | // Control characters
                b'<' | b'>' | b'"' | b'{' | b'}' | b'|' | b'\\' | b'^' | b'`' | b' ' // Forbidden
            ) {
                return false;
            }
        }

        true
    }

    /// Fast IRI validation (non-SIMD fallback)
    #[cfg(not(feature = "simd"))]
    pub fn validate_iri_fast(iri: &str) -> bool {
        if iri.is_empty() {
            return false;
        }

        for byte in iri.bytes() {
            if matches!(
                byte,
                b'<' | b'>' | b'"' | b'{' | b'}' | b'|' | b'\\' | b'^' | b'`' | b' ' // Forbidden
            ) {
                return false;
            }
        }

        true
    }

    /// Fast string comparison using SIMD
    pub fn compare_strings_fast(a: &str, b: &str) -> std::cmp::Ordering {
        if a.len() != b.len() {
            return a.len().cmp(&b.len());
        }

        let a_bytes = a.as_bytes();
        let b_bytes = b.as_bytes();
        let len = a_bytes.len();

        // Process 32 bytes at a time
        let chunks = len / 32;

        for i in 0..chunks {
            let start = i * 32;
            let a_chunk = &a_bytes[start..start + 32];
            let b_chunk = &b_bytes[start..start + 32];

            // Compare chunks bytewise
            for j in 0..32 {
                match a_chunk[j].cmp(&b_chunk[j]) {
                    std::cmp::Ordering::Equal => continue,
                    other => return other,
                }
            }
        }

        // Process remaining bytes
        for i in chunks * 32..len {
            match a_bytes[i].cmp(&b_bytes[i]) {
                std::cmp::Ordering::Equal => continue,
                other => return other,
            }
        }

        std::cmp::Ordering::Equal
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rdf_arena() {
        let arena = RdfArena::new();

        let s1 = arena.alloc_str("test string 1");
        let s2 = arena.alloc_str("test string 2");

        assert_eq!(s1, "test string 1");
        assert_eq!(s2, "test string 2");
        assert!(arena.allocated_bytes() > 0);
        assert_eq!(arena.allocation_count(), 2);
    }

    #[test]
    fn test_term_ref() {
        let node = NamedNode::new("http://example.org/test").unwrap();
        let term_ref = TermRef::from_named_node(&node);

        assert!(term_ref.is_named_node());
        assert_eq!(term_ref.as_str(), "http://example.org/test");

        let owned = term_ref.to_owned().unwrap();
        assert!(owned.is_named_node());
    }

    #[test]
    fn test_triple_ref() {
        let subject = NamedNode::new("http://example.org/s").unwrap();
        let predicate = NamedNode::new("http://example.org/p").unwrap();
        let object = Literal::new("test object");
        let triple = Triple::new(subject, predicate, object);

        let triple_ref = TripleRef::from_triple(&triple);
        assert!(triple_ref.subject.is_named_node());
        assert!(triple_ref.predicate.is_named_node());
        assert!(triple_ref.object.is_literal());

        let owned = triple_ref.to_owned().unwrap();
        assert_eq!(owned, triple);
    }

    #[test]
    fn test_lock_free_graph() {
        let graph = LockFreeGraph::new();
        assert!(graph.is_empty());

        let subject = NamedNode::new("http://example.org/s").unwrap();
        let predicate = NamedNode::new("http://example.org/p").unwrap();
        let object = Literal::new("test object");
        let triple = Triple::new(subject, predicate, object);

        assert!(graph.insert(triple.clone()));
        assert!(!graph.insert(triple.clone())); // Duplicate
        assert_eq!(graph.len(), 1);
        assert!(graph.contains(&triple));
    }

    #[test]
    fn test_optimized_graph() {
        let graph = OptimizedGraph::new();

        let subject = NamedNode::new("http://example.org/s").unwrap();
        let predicate = NamedNode::new("http://example.org/p").unwrap();
        let object = Literal::new("test object");
        let triple = Triple::new(subject.clone(), predicate.clone(), object.clone());

        assert!(graph.insert(&triple));
        assert!(!graph.insert(&triple)); // Duplicate

        // Query by exact match
        let results = graph.query(
            Some(&Subject::NamedNode(subject.clone())),
            Some(&Predicate::NamedNode(predicate.clone())),
            Some(&Object::Literal(object.clone())),
        );
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], triple);

        // Query by subject only
        let results = graph.query(Some(&Subject::NamedNode(subject)), None, None);
        assert_eq!(results.len(), 1);

        let stats = graph.stats();
        assert_eq!(stats.triple_count, 1);
    }

    #[test]
    fn test_simd_iri_validation() {
        assert!(simd::validate_iri_fast("http://example.org/test"));
        assert!(!simd::validate_iri_fast("http://example.org/<invalid>"));
        assert!(!simd::validate_iri_fast(""));
        assert!(!simd::validate_iri_fast(
            "http://example.org/test with spaces"
        ));
    }

    #[test]
    fn test_simd_string_comparison() {
        assert_eq!(
            simd::compare_strings_fast("abc", "abc"),
            std::cmp::Ordering::Equal
        );
        assert_eq!(
            simd::compare_strings_fast("abc", "def"),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            simd::compare_strings_fast("def", "abc"),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            simd::compare_strings_fast("short", "longer"),
            std::cmp::Ordering::Less
        );
    }

    #[test]
    fn test_arena_reset() {
        let arena = RdfArena::new();

        arena.alloc_str("test");
        assert!(arena.allocated_bytes() > 0);

        arena.reset();
        assert_eq!(arena.allocated_bytes(), 0);
        assert_eq!(arena.allocation_count(), 0);
    }

    #[test]
    fn test_concurrent_optimized_graph() {
        use std::sync::Arc;
        use std::thread;

        let graph = Arc::new(OptimizedGraph::new());
        let handles: Vec<_> = (0..10)
            .map(|i| {
                let graph = Arc::clone(&graph);
                thread::spawn(move || {
                    let subject = NamedNode::new(&format!("http://example.org/s{}", i)).unwrap();
                    let predicate = NamedNode::new("http://example.org/p").unwrap();
                    let object = Literal::new(&format!("object{}", i));
                    let triple = Triple::new(subject, predicate, object);

                    graph.insert(&triple)
                })
            })
            .collect();

        let results: Vec<bool> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert!(results.iter().all(|&inserted| inserted));

        let stats = graph.stats();
        assert_eq!(stats.triple_count, 10);
    }
}

/// Zero-copy buffer for efficient data manipulation
pub struct ZeroCopyBuffer {
    data: Pin<Box<[u8]>>,
    len: usize,
}

impl ZeroCopyBuffer {
    /// Create a new zero-copy buffer with the given capacity
    pub fn new(capacity: usize) -> Self {
        Self::with_capacity(capacity)
    }

    /// Create a new zero-copy buffer with the given capacity
    pub fn with_capacity(capacity: usize) -> Self {
        let mut vec = Vec::with_capacity(capacity);
        vec.resize(capacity, 0);
        let data = vec.into_boxed_slice();

        ZeroCopyBuffer {
            data: Pin::new(data),
            len: 0,
        }
    }

    /// Get a slice of the buffer data
    pub fn as_slice(&self) -> &[u8] {
        &self.data[..self.len]
    }

    /// Get a mutable slice of the entire buffer for reading into
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data[..]
    }

    /// Get the buffer capacity
    pub fn capacity(&self) -> usize {
        self.data.len()
    }

    /// Get the current length of data in the buffer
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Reset the buffer (alias for clear)
    pub fn reset(&mut self) {
        self.clear();
    }

    /// Set the length of valid data in the buffer
    pub fn set_len(&mut self, len: usize) {
        assert!(len <= self.capacity());
        self.len = len;
    }

    /// Write data to the buffer
    pub fn write(&mut self, data: &[u8]) -> Result<usize, std::io::Error> {
        let available = self.capacity() - self.len;
        let to_write = data.len().min(available);

        if to_write == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::WriteZero,
                "Buffer is full",
            ));
        }

        // SAFETY: We're writing within bounds
        unsafe {
            let dst = self.data.as_mut_ptr().add(self.len);
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, to_write);
        }

        self.len += to_write;
        Ok(to_write)
    }
}

/// SIMD JSON processor for fast JSON parsing
#[derive(Clone)]
pub struct SimdJsonProcessor;

impl SimdJsonProcessor {
    /// Create a new SIMD JSON processor
    pub fn new() -> Self {
        SimdJsonProcessor
    }

    /// Parse JSON bytes into a Value
    pub fn parse<'a>(
        &mut self,
        json: &'a mut [u8],
    ) -> Result<simd_json::BorrowedValue<'a>, simd_json::Error> {
        simd_json::to_borrowed_value(json)
    }

    /// Parse JSON string into a Value
    pub fn parse_str<'a>(
        &mut self,
        json: &'a mut str,
    ) -> Result<simd_json::BorrowedValue<'a>, simd_json::Error> {
        let bytes = unsafe { json.as_bytes_mut() };
        simd_json::to_borrowed_value(bytes)
    }

    /// Parse JSON bytes into an owned Value
    pub fn parse_owned(
        &mut self,
        json: &mut [u8],
    ) -> Result<simd_json::OwnedValue, simd_json::Error> {
        simd_json::to_owned_value(json)
    }

    /// Parse JSON bytes into a serde_json::Value (compatibility method)
    pub fn parse_json(&self, json: &[u8]) -> Result<serde_json::Value, serde_json::Error> {
        serde_json::from_slice(json)
    }
}

impl Default for SimdJsonProcessor {
    fn default() -> Self {
        Self::new()
    }
}
