//! Ultra-high performance RDF store implementation

use std::collections::{HashMap, HashSet, BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::{Arc, RwLock};
use rayon::prelude::*;
use crate::model::*;
use crate::indexing::{UltraIndex, IndexStats, MemoryUsage};
use crate::optimization::{RdfArena, OptimizedGraph};
use crate::{OxirsError, Result};

// Re-export from oxigraph for compatibility
pub use oxigraph::sparql::QueryResults;

/// Storage backend for RDF quads
#[derive(Debug)]
pub enum StorageBackend {
    /// Ultra-high performance in-memory storage
    UltraMemory(Arc<UltraIndex>, Arc<RdfArena>),
    /// Legacy in-memory storage using collections
    Memory(Arc<RwLock<MemoryStorage>>),
    /// File-based storage (future: will use disk-backed storage)
    Persistent(Arc<RwLock<MemoryStorage>>, std::path::PathBuf),
}

/// In-memory storage implementation
#[derive(Debug, Clone, Default)]
struct MemoryStorage {
    /// All quads in the store
    quads: BTreeSet<Quad>,
    /// Index by subject for efficient lookups
    subject_index: BTreeMap<Subject, BTreeSet<Quad>>,
    /// Index by predicate for efficient lookups
    predicate_index: BTreeMap<Predicate, BTreeSet<Quad>>,
    /// Index by object for efficient lookups
    object_index: BTreeMap<Object, BTreeSet<Quad>>,
    /// Index by graph name for efficient lookups
    graph_index: BTreeMap<GraphName, BTreeSet<Quad>>,
    /// Named graphs in the dataset
    named_graphs: BTreeSet<NamedNode>,
}

impl MemoryStorage {
    fn new() -> Self {
        MemoryStorage {
            quads: BTreeSet::new(),
            subject_index: BTreeMap::new(),
            predicate_index: BTreeMap::new(),
            object_index: BTreeMap::new(),
            graph_index: BTreeMap::new(),
            named_graphs: BTreeSet::new(),
        }
    }
    
    fn insert_quad(&mut self, quad: Quad) -> bool {
        let is_new = self.quads.insert(quad.clone());
        
        if is_new {
            // Update indexes
            self.subject_index
                .entry(quad.subject().clone())
                .or_insert_with(BTreeSet::new)
                .insert(quad.clone());
            
            self.predicate_index
                .entry(quad.predicate().clone())
                .or_insert_with(BTreeSet::new)
                .insert(quad.clone());
            
            self.object_index
                .entry(quad.object().clone())
                .or_insert_with(BTreeSet::new)
                .insert(quad.clone());
            
            self.graph_index
                .entry(quad.graph_name().clone())
                .or_insert_with(BTreeSet::new)
                .insert(quad.clone());
            
            // Add to named graphs if not default graph
            if let GraphName::NamedNode(graph_name) = quad.graph_name() {
                self.named_graphs.insert(graph_name.clone());
            }
        }
        
        is_new
    }
    
    fn remove_quad(&mut self, quad: &Quad) -> bool {
        let was_present = self.quads.remove(quad);
        
        if was_present {
            // Update indexes
            if let Some(subject_quads) = self.subject_index.get_mut(quad.subject()) {
                subject_quads.remove(quad);
                if subject_quads.is_empty() {
                    self.subject_index.remove(quad.subject());
                }
            }
            
            if let Some(predicate_quads) = self.predicate_index.get_mut(quad.predicate()) {
                predicate_quads.remove(quad);
                if predicate_quads.is_empty() {
                    self.predicate_index.remove(quad.predicate());
                }
            }
            
            if let Some(object_quads) = self.object_index.get_mut(quad.object()) {
                object_quads.remove(quad);
                if object_quads.is_empty() {
                    self.object_index.remove(quad.object());
                }
            }
            
            if let Some(graph_quads) = self.graph_index.get_mut(quad.graph_name()) {
                graph_quads.remove(quad);
                if graph_quads.is_empty() {
                    self.graph_index.remove(quad.graph_name());
                    // Remove from named graphs if it was a named graph
                    if let GraphName::NamedNode(graph_name) = quad.graph_name() {
                        self.named_graphs.remove(graph_name);
                    }
                }
            }
        }
        
        was_present
    }
    
    fn contains_quad(&self, quad: &Quad) -> bool {
        self.quads.contains(quad)
    }
    
    fn iter_quads(&self) -> impl Iterator<Item = &Quad> {
        self.quads.iter()
    }
    
    fn query_quads(
        &self,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
        graph_name: Option<&GraphName>,
    ) -> Vec<Quad> {
        // Start with the most selective index
        let mut candidates: Option<BTreeSet<Quad>> = None;
        
        // Use subject index if provided
        if let Some(s) = subject {
            if let Some(subject_quads) = self.subject_index.get(s) {
                candidates = Some(subject_quads.clone());
            } else {
                return Vec::new(); // No quads with this subject
            }
        }
        
        // Intersect with predicate index if provided
        if let Some(p) = predicate {
            if let Some(predicate_quads) = self.predicate_index.get(p) {
                if let Some(ref mut cand) = candidates {
                    *cand = cand.intersection(predicate_quads).cloned().collect();
                } else {
                    candidates = Some(predicate_quads.clone());
                }
            } else {
                return Vec::new(); // No quads with this predicate
            }
        }
        
        // Intersect with object index if provided
        if let Some(o) = object {
            if let Some(object_quads) = self.object_index.get(o) {
                if let Some(ref mut cand) = candidates {
                    *cand = cand.intersection(object_quads).cloned().collect();
                } else {
                    candidates = Some(object_quads.clone());
                }
            } else {
                return Vec::new(); // No quads with this object
            }
        }
        
        // Intersect with graph index if provided
        if let Some(g) = graph_name {
            if let Some(graph_quads) = self.graph_index.get(g) {
                if let Some(ref mut cand) = candidates {
                    *cand = cand.intersection(graph_quads).cloned().collect();
                } else {
                    candidates = Some(graph_quads.clone());
                }
            } else {
                return Vec::new(); // No quads in this graph
            }
        }
        
        // If no specific criteria provided, return all quads
        let quads = candidates.unwrap_or_else(|| self.quads.clone());
        quads.into_iter().collect()
    }
    
    fn len(&self) -> usize {
        self.quads.len()
    }
    
    fn is_empty(&self) -> bool {
        self.quads.is_empty()
    }
}

/// Main RDF store interface
#[derive(Debug)]
pub struct Store {
    backend: StorageBackend,
}

impl Store {
    /// Create a new ultra-high performance in-memory store
    pub fn new() -> Result<Self> {
        Ok(Store {
            backend: StorageBackend::UltraMemory(
                Arc::new(UltraIndex::new()),
                Arc::new(RdfArena::new()),
            ),
        })
    }

    /// Create a new legacy in-memory store for compatibility
    pub fn new_legacy() -> Result<Self> {
        Ok(Store {
            backend: StorageBackend::Memory(Arc::new(RwLock::new(MemoryStorage::new()))),
        })
    }

    /// Create a new persistent store at the given path
    /// 
    /// Note: Currently uses in-memory storage with file path tracking.
    /// Future versions will implement disk-based persistence.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_buf = path.as_ref().to_path_buf();
        Ok(Store {
            backend: StorageBackend::Persistent(
                Arc::new(RwLock::new(MemoryStorage::new())),
                path_buf,
            ),
        })
    }

    /// Insert a quad into the store
    pub fn insert_quad(&mut self, quad: Quad) -> Result<bool> {
        match &self.backend {
            StorageBackend::UltraMemory(index, _arena) => {
                // Check if quad already exists
                let existing = index.find_quads(
                    Some(quad.subject()),
                    Some(quad.predicate()),
                    Some(quad.object()),
                    Some(quad.graph_name())
                );
                if !existing.is_empty() {
                    return Ok(false); // Quad already exists
                }
                
                let _id = index.insert_quad(&quad);
                Ok(true) // New quad inserted
            }
            StorageBackend::Memory(storage) | StorageBackend::Persistent(storage, _) => {
                let mut storage = storage.write().map_err(|e| {
                    OxirsError::Store(format!("Failed to acquire write lock: {}", e))
                })?;
                Ok(storage.insert_quad(quad))
            }
        }
    }

    /// Bulk insert quads for maximum performance
    pub fn bulk_insert_quads(&mut self, quads: Vec<Quad>) -> Result<Vec<u64>> {
        match &self.backend {
            StorageBackend::UltraMemory(index, _arena) => {
                let ids = index.bulk_insert_quads(&quads);
                Ok(ids)
            }
            _ => {
                // Fallback to individual inserts for legacy backend
                let mut ids = Vec::new();
                for quad in quads {
                    self.insert_quad(quad)?;
                    ids.push(0); // Dummy ID for legacy mode
                }
                Ok(ids)
            }
        }
    }
    
    /// Insert a triple into the default graph
    pub fn insert_triple(&mut self, triple: Triple) -> Result<bool> {
        self.insert_quad(Quad::from_triple(triple))
    }
    
    /// Insert a triple into the store (legacy string interface)
    pub fn insert(&mut self, subject: &str, predicate: &str, object: &str) -> Result<bool> {
        let subject_node = NamedNode::new(subject)?;
        let predicate_node = NamedNode::new(predicate)?;
        let object_literal = Literal::new(object);
        
        let triple = Triple::new(subject_node, predicate_node, object_literal);
        self.insert_triple(triple)
    }
    
    /// Remove a quad from the store
    pub fn remove_quad(&mut self, quad: &Quad) -> Result<bool> {
        match &self.backend {
            StorageBackend::UltraMemory(index, _arena) => {
                Ok(index.remove_quad(quad))
            }
            StorageBackend::Memory(storage) | StorageBackend::Persistent(storage, _) => {
                let mut storage = storage.write().map_err(|e| {
                    OxirsError::Store(format!("Failed to acquire write lock: {}", e))
                })?;
                Ok(storage.remove_quad(quad))
            }
        }
    }
    
    /// Check if a quad exists in the store
    pub fn contains_quad(&self, quad: &Quad) -> Result<bool> {
        match &self.backend {
            StorageBackend::Memory(storage) | StorageBackend::Persistent(storage, _) => {
                let storage = storage.read().map_err(|e| {
                    OxirsError::Store(format!("Failed to acquire read lock: {}", e))
                })?;
                Ok(storage.contains_quad(quad))
            }
            StorageBackend::UltraMemory(index, _) => {
                let results = index.find_quads(
                    Some(quad.subject()),
                    Some(quad.predicate()),
                    Some(quad.object()),
                    Some(quad.graph_name())
                );
                Ok(!results.is_empty())
            }
        }
    }
    
    /// Query quads matching the given pattern
    /// 
    /// None values act as wildcards matching any term.
    pub fn query_quads(
        &self,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
        graph_name: Option<&GraphName>,
    ) -> Result<Vec<Quad>> {
        match &self.backend {
            StorageBackend::UltraMemory(index, _arena) => {
                let results = index.find_quads(subject, predicate, object, graph_name);
                Ok(results)
            }
            StorageBackend::Memory(storage) | StorageBackend::Persistent(storage, _) => {
                let storage = storage.read().map_err(|e| {
                    OxirsError::Store(format!("Failed to acquire read lock: {}", e))
                })?;
                Ok(storage.query_quads(subject, predicate, object, graph_name))
            }
        }
    }
    
    /// Query triples in the default graph matching the given pattern
    pub fn query_triples(
        &self,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
    ) -> Result<Vec<Triple>> {
        let default_graph = GraphName::DefaultGraph;
        let quads = self.query_quads(subject, predicate, object, Some(&default_graph))?;
        Ok(quads.into_iter().map(|quad| quad.to_triple()).collect())
    }
    
    /// Get all quads in the store
    pub fn iter_quads(&self) -> Result<Vec<Quad>> {
        self.query_quads(None, None, None, None)
    }
    
    /// Get the number of quads in the store
    pub fn len(&self) -> Result<usize> {
        match &self.backend {
            StorageBackend::UltraMemory(index, _arena) => {
                Ok(index.len())
            }
            StorageBackend::Memory(storage) | StorageBackend::Persistent(storage, _) => {
                let storage = storage.read().map_err(|e| {
                    OxirsError::Store(format!("Failed to acquire read lock: {}", e))
                })?;
                Ok(storage.len())
            }
        }
    }
    
    /// Check if the store is empty
    pub fn is_empty(&self) -> Result<bool> {
        match &self.backend {
            StorageBackend::UltraMemory(index, _arena) => {
                Ok(index.is_empty())
            }
            StorageBackend::Memory(storage) | StorageBackend::Persistent(storage, _) => {
                let storage = storage.read().map_err(|e| {
                    OxirsError::Store(format!("Failed to acquire read lock: {}", e))
                })?;
                Ok(storage.is_empty())
            }
        }
    }

    /// Get performance statistics (ultra-performance mode only)
    pub fn stats(&self) -> Option<Arc<IndexStats>> {
        match &self.backend {
            StorageBackend::UltraMemory(index, _arena) => {
                Some(index.stats())
            }
            _ => None,
        }
    }

    /// Get memory usage statistics (ultra-performance mode only)
    pub fn memory_usage(&self) -> Option<MemoryUsage> {
        match &self.backend {
            StorageBackend::UltraMemory(index, arena) => {
                let mut usage = index.memory_usage();
                usage.arena_bytes = arena.allocated_bytes();
                Some(usage)
            }
            _ => None,
        }
    }

    /// Clear memory arena to reclaim memory (ultra-performance mode only)
    pub fn clear_arena(&self) {
        match &self.backend {
            StorageBackend::UltraMemory(index, _arena) => {
                index.clear_arena();
            }
            _ => {}
        }
    }
    
    /// Clear all data from the store
    pub fn clear(&mut self) -> Result<()> {
        match &self.backend {
            StorageBackend::UltraMemory(index, _arena) => {
                index.clear();
                Ok(())
            }
            StorageBackend::Memory(storage) | StorageBackend::Persistent(storage, _) => {
                let mut storage = storage.write().map_err(|e| {
                    OxirsError::Store(format!("Failed to acquire write lock: {}", e))
                })?;
                *storage = MemoryStorage::new();
                Ok(())
            }
        }
    }

    /// Query the store with SPARQL (placeholder for future implementation)
    pub fn query(&self, sparql: &str) -> Result<OxirsQueryResults> {
        // TODO: Implement SPARQL query execution using spargebra/spareval
        let _sparql = sparql;
        Ok(OxirsQueryResults::new())
    }
}

impl Default for Store {
    fn default() -> Self {
        Store::new().unwrap()
    }
}

/// Query results container (placeholder for future SPARQL implementation)
#[derive(Debug, Clone)]
pub struct OxirsQueryResults {
    // TODO: Implement query results with SPARQL bindings
}

impl OxirsQueryResults {
    pub fn new() -> Self {
        OxirsQueryResults {}
    }
}

impl Default for OxirsQueryResults {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::*;
    
    fn create_test_quad() -> Quad {
        let subject = NamedNode::new("http://example.org/subject").unwrap();
        let predicate = NamedNode::new("http://example.org/predicate").unwrap();
        let object = Literal::new("test object");
        let graph = NamedNode::new("http://example.org/graph").unwrap();
        
        Quad::new(subject, predicate, object, graph)
    }
    
    fn create_test_triple() -> Triple {
        let subject = NamedNode::new("http://example.org/subject").unwrap();
        let predicate = NamedNode::new("http://example.org/predicate").unwrap();
        let object = Literal::new("test object");
        
        Triple::new(subject, predicate, object)
    }
    
    #[test]
    fn test_store_creation() {
        let store = Store::new().unwrap();
        assert!(store.is_empty().unwrap());
        assert_eq!(store.len().unwrap(), 0);
    }
    
    #[test]
    fn test_store_quad_operations() {
        // Use legacy backend for faster testing
        let mut store = Store::new_legacy().unwrap();
        let quad = create_test_quad();
        
        // Test insertion
        assert!(store.insert_quad(quad.clone()).unwrap());
        assert!(!store.is_empty().unwrap());
        assert_eq!(store.len().unwrap(), 1);
        assert!(store.contains_quad(&quad).unwrap());
        
        // Test duplicate insertion
        assert!(!store.insert_quad(quad.clone()).unwrap());
        assert_eq!(store.len().unwrap(), 1);
        
        // Test removal
        assert!(store.remove_quad(&quad).unwrap());
        assert!(store.is_empty().unwrap());
        assert_eq!(store.len().unwrap(), 0);
        assert!(!store.contains_quad(&quad).unwrap());
        
        // Test removal of non-existent quad
        assert!(!store.remove_quad(&quad).unwrap());
    }
    
    #[test]
    fn test_store_triple_operations() {
        let mut store = Store::new().unwrap();
        let triple = create_test_triple();
        
        // Test insertion
        assert!(store.insert_triple(triple.clone()).unwrap());
        assert!(!store.is_empty().unwrap());
        assert_eq!(store.len().unwrap(), 1);
        
        // Verify the triple was inserted in the default graph
        let default_graph = GraphName::DefaultGraph;
        let quads = store.query_quads(None, None, None, Some(&default_graph)).unwrap();
        assert_eq!(quads.len(), 1);
        assert_eq!(quads[0].to_triple(), triple);
    }
    
    #[test]
    fn test_store_string_insertion() {
        let mut store = Store::new().unwrap();
        
        let result = store.insert(
            "http://example.org/subject",
            "http://example.org/predicate", 
            "test object"
        ).unwrap();
        
        assert!(result);
        assert_eq!(store.len().unwrap(), 1);
    }
    
    #[test]
    fn test_store_query_patterns() {
        let mut store = Store::new().unwrap();
        
        // Create test data
        let subject1 = NamedNode::new("http://example.org/subject1").unwrap();
        let subject2 = NamedNode::new("http://example.org/subject2").unwrap();
        let predicate1 = NamedNode::new("http://example.org/predicate1").unwrap();
        let predicate2 = NamedNode::new("http://example.org/predicate2").unwrap();
        let object1 = Literal::new("object1");
        let object2 = Literal::new("object2");
        let graph1 = NamedNode::new("http://example.org/graph1").unwrap();
        let graph2 = NamedNode::new("http://example.org/graph2").unwrap();
        
        let quad1 = Quad::new(subject1.clone(), predicate1.clone(), object1.clone(), graph1.clone());
        let quad2 = Quad::new(subject2.clone(), predicate1.clone(), object2.clone(), graph1.clone());
        let quad3 = Quad::new(subject1.clone(), predicate2.clone(), object1.clone(), graph2.clone());
        
        store.insert_quad(quad1.clone()).unwrap();
        store.insert_quad(quad2.clone()).unwrap();
        store.insert_quad(quad3.clone()).unwrap();
        
        // Test query by subject
        let results = store.query_quads(
            Some(&Subject::NamedNode(subject1.clone())),
            None, None, None
        ).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.contains(&quad1));
        assert!(results.contains(&quad3));
        
        // Test query by predicate
        let results = store.query_quads(
            None,
            Some(&Predicate::NamedNode(predicate1.clone())),
            None, None
        ).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.contains(&quad1));
        assert!(results.contains(&quad2));
        
        // Test query by object
        let results = store.query_quads(
            None, None,
            Some(&Object::Literal(object1.clone())),
            None
        ).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.contains(&quad1));
        assert!(results.contains(&quad3));
        
        // Test query by graph
        let results = store.query_quads(
            None, None, None,
            Some(&GraphName::NamedNode(graph1.clone()))
        ).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.contains(&quad1));
        assert!(results.contains(&quad2));
        
        // Test complex query (subject + predicate)
        let results = store.query_quads(
            Some(&Subject::NamedNode(subject1.clone())),
            Some(&Predicate::NamedNode(predicate1.clone())),
            None, None
        ).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results.contains(&quad1));
        
        // Test query that should return no results
        let non_existent_subject = NamedNode::new("http://example.org/nonexistent").unwrap();
        let results = store.query_quads(
            Some(&Subject::NamedNode(non_existent_subject)),
            None, None, None
        ).unwrap();
        assert_eq!(results.len(), 0);
    }
    
    #[test]
    fn test_store_triple_queries() {
        let mut store = Store::new().unwrap();
        
        // Insert some triples in default graph
        let triple1 = Triple::new(
            NamedNode::new("http://example.org/s1").unwrap(),
            NamedNode::new("http://example.org/p1").unwrap(),
            Literal::new("o1")
        );
        let triple2 = Triple::new(
            NamedNode::new("http://example.org/s2").unwrap(),
            NamedNode::new("http://example.org/p1").unwrap(),
            Literal::new("o2")
        );
        
        store.insert_triple(triple1.clone()).unwrap();
        store.insert_triple(triple2.clone()).unwrap();
        
        // Insert a quad in a named graph (should not appear in triple queries)
        let quad_in_named_graph = Quad::new(
            NamedNode::new("http://example.org/s3").unwrap(),
            NamedNode::new("http://example.org/p1").unwrap(),
            Literal::new("o3"),
            NamedNode::new("http://example.org/namedgraph").unwrap()
        );
        store.insert_quad(quad_in_named_graph).unwrap();
        
        // Query triples by predicate
        let results = store.query_triples(
            None,
            Some(&Predicate::NamedNode(NamedNode::new("http://example.org/p1").unwrap())),
            None
        ).unwrap();
        
        // Should only return triples from default graph
        assert_eq!(results.len(), 2);
        assert!(results.contains(&triple1));
        assert!(results.contains(&triple2));
    }
    
    #[test]
    fn test_store_clear() {
        let mut store = Store::new().unwrap();
        
        // Add some data
        let quad = create_test_quad();
        store.insert_quad(quad).unwrap();
        assert_eq!(store.len().unwrap(), 1);
        
        // Clear the store
        store.clear().unwrap();
        assert!(store.is_empty().unwrap());
        assert_eq!(store.len().unwrap(), 0);
    }
    
    #[test]
    fn test_store_iter_quads() {
        let mut store = Store::new().unwrap();
        
        let quad1 = create_test_quad();
        let quad2 = Quad::new(
            NamedNode::new("http://example.org/subject").unwrap(),
            NamedNode::new("http://example.org/predicate").unwrap(),
            Literal::new("different object"),
            NamedNode::new("http://example.org/graph").unwrap()
        );
        
        store.insert_quad(quad1.clone()).unwrap();
        store.insert_quad(quad2.clone()).unwrap();
        
        let all_quads = store.iter_quads().unwrap();
        assert_eq!(all_quads.len(), 2);
        assert!(all_quads.contains(&quad1));
        assert!(all_quads.contains(&quad2));
    }
    
    #[test]
    fn test_persistent_store_creation() {
        use std::env;
        
        let temp_path = env::temp_dir().join("oxirs_test_store");
        let store = Store::open(&temp_path).unwrap();
        
        // Should start empty like memory store
        assert!(store.is_empty().unwrap());
        assert_eq!(store.len().unwrap(), 0);
        
        // Verify backend type (though both currently use memory storage)
        match store.backend {
            StorageBackend::Persistent(_, path) => {
                assert_eq!(path, temp_path);
            }
            _ => panic!("Expected persistent backend"),
        }
    }
    
    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;
        use std::thread;
        
        let store = Arc::new(Store::new().unwrap());
        let store_clone = Arc::clone(&store);
        
        // Spawn a thread that reads from the store
        let reader_handle = thread::spawn(move || {
            for _ in 0..100 {
                let _ = store_clone.is_empty().unwrap();
                let _ = store_clone.len().unwrap();
            }
        });
        
        // Read from the main thread as well
        for _ in 0..100 {
            let _ = store.is_empty().unwrap();
            let _ = store.len().unwrap();
        }
        
        reader_handle.join().unwrap();
    }
    
    #[test]
    fn test_basic_workflow() {
        use crate::serializer::Serializer;
        use crate::parser::RdfFormat;
        use crate::model::graph::Graph;
        
        let mut store = Store::new().unwrap();
        
        // Add some test data
        let subject = NamedNode::new("http://example.org/person/alice").unwrap();
        let name_pred = NamedNode::new("http://xmlns.com/foaf/0.1/name").unwrap();
        let age_pred = NamedNode::new("http://xmlns.com/foaf/0.1/age").unwrap();
        let name_obj = Literal::new("Alice Smith");
        let age_obj = Literal::new_typed("30", NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap());
        
        let triple1 = Triple::new(subject.clone(), name_pred, name_obj);
        let triple2 = Triple::new(subject.clone(), age_pred, age_obj);
        
        store.insert_triple(triple1.clone()).unwrap();
        store.insert_triple(triple2.clone()).unwrap();
        
        // Test basic store operations
        assert_eq!(store.len().unwrap(), 2);
        assert!(!store.is_empty().unwrap());
        
        // Test serialization with N-Triples (which is implemented)
        let serializer = Serializer::new(RdfFormat::NTriples);
        let graph = Graph::from_iter(vec![triple1, triple2]);
        let ntriples_output = serializer.serialize_graph(&graph).unwrap();
        assert!(!ntriples_output.is_empty());
        assert!(ntriples_output.contains("Alice Smith"));
    }
}