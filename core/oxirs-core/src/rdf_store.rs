//! Ultra-high performance RDF store implementation

use crate::indexing::{IndexStats, MemoryUsage, UltraIndex};
use crate::model::*;
use crate::optimization::RdfArena;
use crate::{OxirsError, Result};
use async_trait::async_trait;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::Path;
use std::sync::{Arc, RwLock};

/// SPARQL query results supporting different result types
#[derive(Debug, Clone)]
pub enum QueryResults {
    /// SELECT query results - variable bindings
    Bindings(Vec<VariableBinding>),
    /// ASK query results - boolean
    Boolean(bool),
    /// CONSTRUCT/DESCRIBE query results - RDF quads
    Graph(Vec<Quad>),
}

impl QueryResults {
    /// Create empty SELECT results
    pub fn empty_bindings() -> Self {
        QueryResults::Bindings(Vec::new())
    }

    /// Create ASK result
    pub fn boolean(value: bool) -> Self {
        QueryResults::Boolean(value)
    }

    /// Create CONSTRUCT/DESCRIBE results
    pub fn graph(quads: Vec<Quad>) -> Self {
        QueryResults::Graph(quads)
    }

    /// Check if results are empty
    pub fn is_empty(&self) -> bool {
        match self {
            QueryResults::Bindings(bindings) => bindings.is_empty(),
            QueryResults::Boolean(_) => false,
            QueryResults::Graph(quads) => quads.is_empty(),
        }
    }

    /// Get the number of results
    pub fn len(&self) -> usize {
        match self {
            QueryResults::Bindings(bindings) => bindings.len(),
            QueryResults::Boolean(_) => 1,
            QueryResults::Graph(quads) => quads.len(),
        }
    }
}

/// Variable binding for SELECT query results
#[derive(Debug, Clone)]
pub struct VariableBinding {
    bindings: std::collections::HashMap<String, Term>,
}

impl VariableBinding {
    pub fn new() -> Self {
        Self {
            bindings: std::collections::HashMap::new(),
        }
    }

    pub fn bind(&mut self, variable: String, value: Term) {
        self.bindings.insert(variable, value);
    }

    pub fn get(&self, variable: &str) -> Option<&Term> {
        self.bindings.get(variable)
    }

    pub fn variables(&self) -> impl Iterator<Item = &String> {
        self.bindings.keys()
    }

    pub fn values(&self) -> impl Iterator<Item = &Term> {
        self.bindings.values()
    }
}

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

/// Store trait for RDF operations
#[async_trait]
pub trait Store: Send + Sync {
    /// Insert a quad into the store
    fn insert_quad(&mut self, quad: Quad) -> Result<bool>;

    /// Remove a quad from the store  
    fn remove_quad(&mut self, quad: &Quad) -> Result<bool>;

    /// Find quads matching the given pattern
    fn find_quads(
        &self,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
        graph_name: Option<&GraphName>,
    ) -> Result<Vec<Quad>>;

    /// Check if the store is ready for operations
    fn is_ready(&self) -> bool;

    /// Get the number of quads in the store
    fn len(&self) -> Result<usize>;

    /// Check if the store is empty
    fn is_empty(&self) -> Result<bool>;

    /// Query the store with SPARQL
    fn query(&self, sparql: &str) -> Result<OxirsQueryResults>;

    /// Prepare a SPARQL query for execution
    fn prepare_query(&self, sparql: &str) -> Result<PreparedQuery>;

    /// Get all triples in the store (converts quads to triples)
    fn triples(&self) -> Result<Vec<Triple>> {
        let quads = self.find_quads(None, None, None, None)?;
        Ok(quads
            .into_iter()
            .map(|quad| {
                Triple::new(
                    quad.subject().clone(),
                    quad.predicate().clone(),
                    quad.object().clone(),
                )
            })
            .collect())
    }
}

/// Prepared SPARQL query
pub struct PreparedQuery {
    sparql: String,
}

impl PreparedQuery {
    pub fn new(sparql: String) -> Self {
        Self { sparql }
    }

    /// Execute the prepared query
    pub fn exec(&self) -> Result<QueryResultsIterator> {
        // Simplified implementation - in reality this would parse and execute SPARQL
        Ok(QueryResultsIterator::empty())
    }
}

/// Iterator over query results
pub struct QueryResultsIterator {
    results: Vec<SolutionMapping>,
    index: usize,
}

impl QueryResultsIterator {
    pub fn empty() -> Self {
        Self {
            results: Vec::new(),
            index: 0,
        }
    }
}

impl Iterator for QueryResultsIterator {
    type Item = SolutionMapping;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.results.len() {
            let result = self.results[self.index].clone();
            self.index += 1;
            Some(result)
        } else {
            None
        }
    }
}

/// Solution mapping for SPARQL query results
#[derive(Debug, Clone)]
pub struct SolutionMapping {
    bindings: std::collections::HashMap<String, Term>,
}

impl SolutionMapping {
    pub fn new() -> Self {
        Self {
            bindings: std::collections::HashMap::new(),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &Term)> {
        self.bindings.iter()
    }
}

/// Main RDF store implementation
#[derive(Debug)]
pub struct RdfStore {
    backend: StorageBackend,
}

impl RdfStore {
    /// Create a new ultra-high performance in-memory store
    pub fn new() -> Result<Self> {
        Ok(RdfStore {
            backend: StorageBackend::UltraMemory(
                Arc::new(UltraIndex::new()),
                Arc::new(RdfArena::new()),
            ),
        })
    }

    /// Create a new legacy in-memory store for compatibility
    pub fn new_legacy() -> Result<Self> {
        Ok(RdfStore {
            backend: StorageBackend::Memory(Arc::new(RwLock::new(MemoryStorage::new()))),
        })
    }

    /// Create a new persistent store at the given path
    ///
    /// Note: Currently uses in-memory storage with file path tracking.
    /// Future versions will implement disk-based persistence.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_buf = path.as_ref().to_path_buf();
        Ok(RdfStore {
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
                    Some(quad.graph_name()),
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
    pub fn insert_string_triple(
        &mut self,
        subject: &str,
        predicate: &str,
        object: &str,
    ) -> Result<bool> {
        let subject_node = NamedNode::new(subject)?;
        let predicate_node = NamedNode::new(predicate)?;
        let object_literal = Literal::new(object);

        let triple = Triple::new(subject_node, predicate_node, object_literal);
        self.insert_triple(triple)
    }

    /// Remove a quad from the store
    pub fn remove_quad(&mut self, quad: &Quad) -> Result<bool> {
        match &self.backend {
            StorageBackend::UltraMemory(index, _arena) => Ok(index.remove_quad(quad)),
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
                    Some(quad.graph_name()),
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

    /// Get all triples in the default graph
    pub fn triples(&self) -> Result<Vec<Triple>> {
        self.query_triples(None, None, None)
    }

    /// Get the number of quads in the store
    pub fn len(&self) -> Result<usize> {
        match &self.backend {
            StorageBackend::UltraMemory(index, _arena) => Ok(index.len()),
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
            StorageBackend::UltraMemory(index, _arena) => Ok(index.is_empty()),
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
            StorageBackend::UltraMemory(index, _arena) => Some(index.stats()),
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

    /// Insert a quad (compatibility alias for insert_quad)
    pub fn insert(&mut self, quad: &Quad) -> Result<()> {
        self.insert_quad(quad.clone())?;
        Ok(())
    }

    /// Remove a quad (compatibility alias for remove_quad)
    pub fn remove(&mut self, quad: &Quad) -> Result<bool> {
        self.remove_quad(quad)
    }

    /// Get all quads in the store
    pub fn quads(&self) -> Result<Vec<Quad>> {
        self.iter_quads()
    }

    /// Get all quads from named graphs only
    pub fn named_graph_quads(&self) -> Result<Vec<Quad>> {
        match &self.backend {
            StorageBackend::UltraMemory(index, _arena) => {
                let mut result = Vec::new();
                // Get all named graphs
                let graphs = self.named_graphs()?;
                for graph in graphs {
                    let graph_name = GraphName::NamedNode(graph);
                    let quads = index.find_quads(None, None, None, Some(&graph_name));
                    result.extend(quads);
                }
                Ok(result)
            }
            StorageBackend::Memory(storage) | StorageBackend::Persistent(storage, _) => {
                let storage = storage.read().map_err(|e| {
                    OxirsError::Store(format!("Failed to acquire read lock: {}", e))
                })?;
                let mut result = Vec::new();
                for graph in &storage.named_graphs {
                    let graph_name = GraphName::NamedNode(graph.clone());
                    let quads = storage.query_quads(None, None, None, Some(&graph_name));
                    result.extend(quads);
                }
                Ok(result)
            }
        }
    }

    /// Get all quads from the default graph only
    pub fn default_graph_quads(&self) -> Result<Vec<Quad>> {
        let default_graph = GraphName::DefaultGraph;
        self.query_quads(None, None, None, Some(&default_graph))
    }

    /// Get quads from a specific graph
    pub fn graph_quads(&self, graph: Option<&NamedNode>) -> Result<Vec<Quad>> {
        let graph_name = graph
            .map(|g| GraphName::NamedNode(g.clone()))
            .unwrap_or(GraphName::DefaultGraph);
        self.query_quads(None, None, None, Some(&graph_name))
    }

    /// Clear all data from all graphs
    pub fn clear_all(&mut self) -> Result<usize> {
        let count = self.len()?;
        self.clear()?;
        Ok(count)
    }

    /// Clear all named graphs (but not the default graph)
    pub fn clear_named_graphs(&mut self) -> Result<usize> {
        let mut deleted = 0;
        let graphs = self.named_graphs()?;

        for graph in graphs {
            let graph_name = GraphName::NamedNode(graph);
            deleted += self.clear_graph(Some(&graph_name))?;
        }

        Ok(deleted)
    }

    /// Clear the default graph only
    pub fn clear_default_graph(&mut self) -> Result<usize> {
        self.clear_graph(None)
    }

    /// Clear a specific graph
    pub fn clear_graph(&mut self, graph: Option<&GraphName>) -> Result<usize> {
        let graph_name = graph.cloned().unwrap_or(GraphName::DefaultGraph);
        let quads = self.query_quads(None, None, None, Some(&graph_name))?;
        let count = quads.len();

        for quad in quads {
            self.remove_quad(&quad)?;
        }

        Ok(count)
    }

    /// Get all graphs (including default if it contains data)
    pub fn graphs(&self) -> Result<Vec<NamedNode>> {
        let mut graphs = self.named_graphs()?;

        // Check if default graph has any data
        let default_graph = GraphName::DefaultGraph;
        let default_quads = self.query_quads(None, None, None, Some(&default_graph))?;
        if !default_quads.is_empty() {
            // Add a special marker for default graph
            if let Ok(default_marker) = NamedNode::new("urn:x-oxirs:default-graph") {
                graphs.push(default_marker);
            }
        }

        Ok(graphs)
    }

    /// Get all named graphs
    pub fn named_graphs(&self) -> Result<Vec<NamedNode>> {
        match &self.backend {
            StorageBackend::UltraMemory(index, _arena) => {
                // Get unique graph names from all quads
                let mut graphs = HashSet::new();
                let all_quads = index.find_quads(None, None, None, None);
                for quad in all_quads {
                    if let GraphName::NamedNode(graph) = quad.graph_name() {
                        graphs.insert(graph.clone());
                    }
                }
                Ok(graphs.into_iter().collect())
            }
            StorageBackend::Memory(storage) | StorageBackend::Persistent(storage, _) => {
                let storage = storage.read().map_err(|e| {
                    OxirsError::Store(format!("Failed to acquire read lock: {}", e))
                })?;
                Ok(storage.named_graphs.iter().cloned().collect())
            }
        }
    }

    /// Create a new graph (if it doesn't exist)
    pub fn create_graph(&mut self, graph: Option<&NamedNode>) -> Result<()> {
        if let Some(graph_name) = graph {
            match &self.backend {
                StorageBackend::Memory(storage) | StorageBackend::Persistent(storage, _) => {
                    let mut storage = storage.write().map_err(|e| {
                        OxirsError::Store(format!("Failed to acquire write lock: {}", e))
                    })?;
                    storage.named_graphs.insert(graph_name.clone());
                }
                StorageBackend::UltraMemory(_, _) => {
                    // Graphs are created implicitly when quads are added
                }
            }
        }
        Ok(())
    }

    /// Drop a graph (remove the graph and all its quads)
    pub fn drop_graph(&mut self, graph: Option<&GraphName>) -> Result<()> {
        self.clear_graph(graph)?;

        // Remove from named graphs set if it's a named graph
        if let Some(GraphName::NamedNode(graph_name)) = graph {
            match &self.backend {
                StorageBackend::Memory(storage) | StorageBackend::Persistent(storage, _) => {
                    let mut storage = storage.write().map_err(|e| {
                        OxirsError::Store(format!("Failed to acquire write lock: {}", e))
                    })?;
                    storage.named_graphs.remove(graph_name);
                }
                StorageBackend::UltraMemory(_, _) => {
                    // Graph is dropped implicitly when all quads are removed
                }
            }
        }

        Ok(())
    }

    /// Load data from a URL into a graph
    pub fn load_from_url(&mut self, url: &str, graph: Option<&NamedNode>) -> Result<usize> {
        // TODO: Implement HTTP fetching and parsing
        // For now, return an error
        Err(OxirsError::Store(format!(
            "Loading from URL not yet implemented: {}",
            url
        )))
    }
}

impl Default for RdfStore {
    fn default() -> Self {
        RdfStore::new().unwrap()
    }
}

// Implement the Store trait for RdfStore
#[async_trait]
impl Store for RdfStore {
    fn insert_quad(&mut self, quad: Quad) -> Result<bool> {
        self.insert_quad(quad)
    }

    fn remove_quad(&mut self, quad: &Quad) -> Result<bool> {
        self.remove_quad(quad)
    }

    fn find_quads(
        &self,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
        graph_name: Option<&GraphName>,
    ) -> Result<Vec<Quad>> {
        self.query_quads(subject, predicate, object, graph_name)
    }

    fn is_ready(&self) -> bool {
        true // Simple implementation
    }

    fn len(&self) -> Result<usize> {
        self.len()
    }

    fn is_empty(&self) -> Result<bool> {
        self.is_empty()
    }

    fn query(&self, sparql: &str) -> Result<OxirsQueryResults> {
        self.query(sparql)
    }

    fn prepare_query(&self, sparql: &str) -> Result<PreparedQuery> {
        Ok(PreparedQuery::new(sparql.to_string()))
    }
}

// ConcreteStore struct for external use with easy construction
#[derive(Debug)]
pub struct ConcreteStore {
    inner: RdfStore,
}

impl ConcreteStore {
    pub fn new() -> Result<Self> {
        Ok(ConcreteStore {
            inner: RdfStore::new()?,
        })
    }
}

impl Default for ConcreteStore {
    fn default() -> Self {
        ConcreteStore::new().unwrap()
    }
}

// Implement Store trait for ConcreteStore
#[async_trait]
impl Store for ConcreteStore {
    fn insert_quad(&mut self, quad: Quad) -> Result<bool> {
        self.inner.insert_quad(quad)
    }

    fn remove_quad(&mut self, quad: &Quad) -> Result<bool> {
        self.inner.remove_quad(quad)
    }

    fn find_quads(
        &self,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
        graph_name: Option<&GraphName>,
    ) -> Result<Vec<Quad>> {
        self.inner
            .find_quads(subject, predicate, object, graph_name)
    }

    fn is_ready(&self) -> bool {
        self.inner.is_ready()
    }

    fn len(&self) -> Result<usize> {
        self.inner.len()
    }

    fn is_empty(&self) -> Result<bool> {
        self.inner.is_empty()
    }

    fn query(&self, sparql: &str) -> Result<OxirsQueryResults> {
        self.inner.query(sparql)
    }

    fn prepare_query(&self, sparql: &str) -> Result<PreparedQuery> {
        self.inner.prepare_query(sparql)
    }
}

/// Query results container for SPARQL queries
#[derive(Debug, Clone)]
pub struct OxirsQueryResults {
    results: QueryResults,
    variables: Vec<String>,
}

impl OxirsQueryResults {
    pub fn new() -> Self {
        OxirsQueryResults {
            results: QueryResults::empty_bindings(),
            variables: Vec::new(),
        }
    }

    pub fn from_bindings(bindings: Vec<VariableBinding>, variables: Vec<String>) -> Self {
        OxirsQueryResults {
            results: QueryResults::Bindings(bindings),
            variables,
        }
    }

    pub fn from_boolean(value: bool) -> Self {
        OxirsQueryResults {
            results: QueryResults::Boolean(value),
            variables: Vec::new(),
        }
    }

    pub fn from_graph(quads: Vec<Quad>) -> Self {
        OxirsQueryResults {
            results: QueryResults::Graph(quads),
            variables: Vec::new(),
        }
    }

    pub fn results(&self) -> &QueryResults {
        &self.results
    }

    pub fn variables(&self) -> &[String] {
        &self.variables
    }

    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    pub fn len(&self) -> usize {
        self.results.len()
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
        let store = RdfStore::new().unwrap();
        assert!(store.is_empty().unwrap());
        assert_eq!(store.len().unwrap(), 0);
    }

    #[test]
    fn test_store_quad_operations() {
        // Use legacy backend for faster testing
        let mut store = RdfStore::new_legacy().unwrap();
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
        let mut store = RdfStore::new().unwrap();
        let triple = create_test_triple();

        // Test insertion
        assert!(store.insert_triple(triple.clone()).unwrap());
        assert!(!store.is_empty().unwrap());
        assert_eq!(store.len().unwrap(), 1);

        // Verify the triple was inserted in the default graph
        let default_graph = GraphName::DefaultGraph;
        let quads = store
            .query_quads(None, None, None, Some(&default_graph))
            .unwrap();
        assert_eq!(quads.len(), 1);
        assert_eq!(quads[0].to_triple(), triple);
    }

    #[test]
    fn test_store_string_insertion() {
        let mut store = RdfStore::new().unwrap();

        let result = store
            .insert_string_triple(
                "http://example.org/subject",
                "http://example.org/predicate",
                "test object",
            )
            .unwrap();

        assert!(result);
        assert_eq!(store.len().unwrap(), 1);
    }

    #[test]
    fn test_store_query_patterns() {
        let mut store = RdfStore::new().unwrap();

        // Create test data
        let subject1 = NamedNode::new("http://example.org/subject1").unwrap();
        let subject2 = NamedNode::new("http://example.org/subject2").unwrap();
        let predicate1 = NamedNode::new("http://example.org/predicate1").unwrap();
        let predicate2 = NamedNode::new("http://example.org/predicate2").unwrap();
        let object1 = Literal::new("object1");
        let object2 = Literal::new("object2");
        let graph1 = NamedNode::new("http://example.org/graph1").unwrap();
        let graph2 = NamedNode::new("http://example.org/graph2").unwrap();

        let quad1 = Quad::new(
            subject1.clone(),
            predicate1.clone(),
            object1.clone(),
            graph1,
        );
        let quad2 = Quad::new(
            subject1.clone(),
            predicate2.clone(),
            object2.clone(),
            graph2.clone(),
        );
        let quad3 = Quad::new(
            subject2,
            predicate1.clone(),
            object2.clone(),
            graph2.clone(),
        );

        // Insert test data
        store.insert_quad(quad1).unwrap();
        store.insert_quad(quad2).unwrap();
        store.insert_quad(quad3).unwrap();

        // Test query by subject
        let s = Subject::NamedNode(subject1);
        let results = store.query_quads(Some(&s), None, None, None).unwrap();
        assert_eq!(results.len(), 2);

        // Test query by predicate
        let p = Predicate::from(predicate1);
        let results = store.query_quads(None, Some(&p), None, None).unwrap();
        assert_eq!(results.len(), 2);

        // Test query by object
        let o = Object::Literal(object2);
        let results = store.query_quads(None, None, Some(&o), None).unwrap();
        assert_eq!(results.len(), 2);

        // Test query by graph
        let g = GraphName::NamedNode(graph2);
        let results = store.query_quads(None, None, None, Some(&g)).unwrap();
        assert_eq!(results.len(), 2);

        // Test combined query
        let results = store.query_quads(Some(&s), Some(&p), None, None).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_store_clear() {
        let mut store = RdfStore::new().unwrap();

        // Insert some data
        for i in 0..5 {
            let subject = NamedNode::new(format!("http://example.org/subject{}", i)).unwrap();
            let predicate = NamedNode::new("http://example.org/predicate").unwrap();
            let object = Literal::new(format!("object{}", i));

            let triple = Triple::new(subject, predicate, object);
            store.insert_triple(triple).unwrap();
        }

        assert_eq!(store.len().unwrap(), 5);

        // Clear the store
        store.clear().unwrap();
        assert!(store.is_empty().unwrap());
        assert_eq!(store.len().unwrap(), 0);
    }

    #[test]
    fn test_bulk_insert() {
        let mut store = RdfStore::new().unwrap();

        let mut quads = Vec::new();
        for i in 0..100 {
            let subject = NamedNode::new(format!("http://example.org/subject{}", i)).unwrap();
            let predicate = NamedNode::new("http://example.org/predicate").unwrap();
            let object = Literal::new(format!("object{}", i));
            let graph = NamedNode::new("http://example.org/graph").unwrap();

            quads.push(Quad::new(subject, predicate, object, graph));
        }

        let ids = store.bulk_insert_quads(quads).unwrap();
        assert_eq!(ids.len(), 100);
        assert_eq!(store.len().unwrap(), 100);
    }

    #[test]
    fn test_default_graph_operations() {
        let mut store = RdfStore::new().unwrap();

        // Insert into default graph
        let triple = create_test_triple();
        store.insert_triple(triple).unwrap();

        // Insert into named graph
        let quad = create_test_quad();
        store.insert_quad(quad).unwrap();

        // Query default graph
        let default_triples = store.query_triples(None, None, None).unwrap();
        assert_eq!(default_triples.len(), 1);

        // Query all quads
        let all_quads = store.iter_quads().unwrap();
        assert_eq!(all_quads.len(), 2);
    }
}
