//! RDF store implementation with pluggable storage backends

pub mod concrete;
pub mod storage;
pub mod types;

pub use concrete::*;
pub use storage::*;
pub use types::*;

use crate::indexing::{IndexStats, MemoryUsage};
use crate::model::*;
use crate::parser::RdfFormat;
use crate::serializer::Serializer;
use crate::sparql::extract_and_expand_prefixes; // SPARQL execution engine
use crate::{OxirsError, Result};
use async_trait::async_trait;
use std::collections::HashSet;
use std::path::Path;
use std::sync::{Arc, RwLock};

/// Store trait for RDF operations
#[async_trait]
pub trait Store: Send + Sync {
    /// Insert a quad into the store
    fn insert_quad(&self, quad: Quad) -> Result<bool>;

    /// Remove a quad from the store  
    fn remove_quad(&self, quad: &Quad) -> Result<bool>;

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

    /// Insert a triple into the default graph
    fn insert_triple(&self, triple: Triple) -> Result<bool> {
        let quad = Quad::from_triple(triple);
        self.insert_quad(quad)
    }

    /// Insert a quad (compatibility method)
    fn insert(&self, quad: &Quad) -> Result<()> {
        self.insert_quad(quad.clone())?;
        Ok(())
    }

    /// Remove a quad (compatibility method)
    fn remove(&self, quad: &Quad) -> Result<bool> {
        self.remove_quad(quad)
    }

    /// Get all quads in the store
    fn quads(&self) -> Result<Vec<Quad>> {
        self.find_quads(None, None, None, None)
    }

    /// Get all named graphs
    fn named_graphs(&self) -> Result<Vec<NamedNode>> {
        // Default implementation - subclasses should override
        Ok(Vec::new())
    }

    /// Get all graphs
    fn graphs(&self) -> Result<Vec<NamedNode>> {
        self.named_graphs()
    }

    /// Get quads from named graphs only
    fn named_graph_quads(&self) -> Result<Vec<Quad>> {
        // Default implementation - get all quads except default graph
        let all_quads = self.quads()?;
        Ok(all_quads
            .into_iter()
            .filter(|quad| matches!(quad.graph_name(), GraphName::NamedNode(_)))
            .collect())
    }

    /// Get quads from the default graph only
    fn default_graph_quads(&self) -> Result<Vec<Quad>> {
        let default_graph = GraphName::DefaultGraph;
        self.find_quads(None, None, None, Some(&default_graph))
    }

    /// Get quads from a specific graph
    fn graph_quads(&self, graph: Option<&NamedNode>) -> Result<Vec<Quad>> {
        let graph_name = graph
            .map(|g| GraphName::NamedNode(g.clone()))
            .unwrap_or(GraphName::DefaultGraph);
        self.find_quads(None, None, None, Some(&graph_name))
    }

    /// Clear all data from all graphs
    fn clear_all(&self) -> Result<usize> {
        // Default implementation - not supported in trait
        Err(OxirsError::NotSupported(
            "clear_all requires mutable access".to_string(),
        ))
    }

    /// Clear all named graphs (but not the default graph)
    fn clear_named_graphs(&self) -> Result<usize> {
        // Default implementation - not supported in trait
        Err(OxirsError::NotSupported(
            "clear_named_graphs requires mutable access".to_string(),
        ))
    }

    /// Clear the default graph only
    fn clear_default_graph(&self) -> Result<usize> {
        // Default implementation - not supported in trait
        Err(OxirsError::NotSupported(
            "clear_default_graph requires mutable access".to_string(),
        ))
    }

    /// Clear a specific graph
    fn clear_graph(&self, _graph: Option<&GraphName>) -> Result<usize> {
        // Default implementation - not supported in trait
        Err(OxirsError::NotSupported(
            "clear_graph requires mutable access".to_string(),
        ))
    }

    /// Create a new graph (if it doesn't exist)
    fn create_graph(&self, _graph: Option<&NamedNode>) -> Result<()> {
        // Default implementation - not supported in trait
        Err(OxirsError::NotSupported(
            "create_graph requires mutable access".to_string(),
        ))
    }

    /// Drop a graph (remove the graph and all its quads)
    fn drop_graph(&self, _graph: Option<&GraphName>) -> Result<()> {
        // Default implementation - not supported in trait
        Err(OxirsError::NotSupported(
            "drop_graph requires mutable access".to_string(),
        ))
    }

    /// Load data from a URL into a graph
    fn load_from_url(&self, _url: &str, _graph: Option<&NamedNode>) -> Result<usize> {
        // Default implementation - not supported in trait
        Err(OxirsError::NotSupported(
            "load_from_url requires mutable access".to_string(),
        ))
    }

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
    #[allow(dead_code)]
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
#[derive(Debug, Clone, Default)]
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
            backend: StorageBackend::Memory(Arc::new(RwLock::new(MemoryStorage::new()))),
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
    /// Loads existing data from disk if present, otherwise creates a new store.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_buf = path.as_ref().to_path_buf();
        let data_file = path_buf.join("data.nq");

        let storage = if data_file.exists() {
            // Load existing data from disk
            Self::load_from_disk(&data_file)?
        } else {
            MemoryStorage::new()
        };

        Ok(RdfStore {
            backend: StorageBackend::Persistent(Arc::new(RwLock::new(storage)), path_buf),
        })
    }

    /// Load data from disk (N-Quads format)
    fn load_from_disk(data_file: &Path) -> Result<MemoryStorage> {
        use crate::format::{RdfFormat, RdfParser};
        use std::io::BufReader;

        let mut storage = MemoryStorage::new();

        if let Ok(file) = std::fs::File::open(data_file) {
            let reader = BufReader::new(file);
            let parser = RdfParser::new(RdfFormat::NQuads);

            for quad in parser.for_reader(reader).flatten() {
                storage.insert_quad(quad);
            }
        }

        Ok(storage)
    }

    /// Save data to disk (N-Quads format)
    fn save_to_disk(&self) -> Result<()> {
        if let StorageBackend::Persistent(storage, path) = &self.backend {
            use std::io::Write;

            let data_file = path.join("data.nq");

            // Ensure directory exists
            if let Some(parent) = data_file.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| OxirsError::Store(format!("Failed to create directory: {e}")))?;
            }

            let storage_guard = storage
                .read()
                .map_err(|e| OxirsError::Store(format!("Failed to acquire read lock: {e}")))?;

            // Create file
            let mut file = std::fs::File::create(&data_file)
                .map_err(|e| OxirsError::Store(format!("Failed to create data file: {e}")))?;

            // Write each quad as N-Quads line
            let serializer = Serializer::new(RdfFormat::NQuads);
            for quad in &storage_guard.quads {
                let line = serializer.serialize_quad_to_nquads(quad)?;
                writeln!(file, "{}", line)
                    .map_err(|e| OxirsError::Store(format!("Failed to write quad: {e}")))?;
            }
        }

        Ok(())
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
                let mut storage = storage
                    .write()
                    .map_err(|e| OxirsError::Store(format!("Failed to acquire write lock: {e}")))?;
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
                let mut storage = storage
                    .write()
                    .map_err(|e| OxirsError::Store(format!("Failed to acquire write lock: {e}")))?;
                Ok(storage.remove_quad(quad))
            }
        }
    }

    /// Check if a quad exists in the store
    pub fn contains_quad(&self, quad: &Quad) -> Result<bool> {
        match &self.backend {
            StorageBackend::Memory(storage) | StorageBackend::Persistent(storage, _) => {
                let storage = storage
                    .read()
                    .map_err(|e| OxirsError::Store(format!("Failed to acquire read lock: {e}")))?;
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
                let storage = storage
                    .read()
                    .map_err(|e| OxirsError::Store(format!("Failed to acquire read lock: {e}")))?;
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
                let storage = storage
                    .read()
                    .map_err(|e| OxirsError::Store(format!("Failed to acquire read lock: {e}")))?;
                Ok(storage.len())
            }
        }
    }

    /// Check if the store is empty
    pub fn is_empty(&self) -> Result<bool> {
        match &self.backend {
            StorageBackend::UltraMemory(index, _arena) => Ok(index.is_empty()),
            StorageBackend::Memory(storage) | StorageBackend::Persistent(storage, _) => {
                let storage = storage
                    .read()
                    .map_err(|e| OxirsError::Store(format!("Failed to acquire read lock: {e}")))?;
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
        if let StorageBackend::UltraMemory(index, _arena) = &self.backend {
            index.clear_arena();
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
                let mut storage = storage
                    .write()
                    .map_err(|e| OxirsError::Store(format!("Failed to acquire write lock: {e}")))?;
                *storage = MemoryStorage::new();
                Ok(())
            }
        }
    }

    /// Extract PREFIX declarations and expand prefixed names in the query
    #[allow(dead_code)]
    fn extract_and_expand_prefixes(
        &self,
        sparql: &str,
    ) -> Result<(std::collections::HashMap<String, String>, String)> {
        extract_and_expand_prefixes(sparql)
    }

    /// Query the store with SPARQL (delegates to QueryExecutor)
    pub fn query(&self, sparql: &str) -> Result<OxirsQueryResults> {
        let executor = crate::sparql::QueryExecutor::new(&self.backend);
        executor.execute(sparql)
    }

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
                let storage = storage
                    .read()
                    .map_err(|e| OxirsError::Store(format!("Failed to acquire read lock: {e}")))?;
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
                let storage = storage
                    .read()
                    .map_err(|e| OxirsError::Store(format!("Failed to acquire read lock: {e}")))?;
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
                        OxirsError::Store(format!("Failed to acquire write lock: {e}"))
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
                        OxirsError::Store(format!("Failed to acquire write lock: {e}"))
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
        use crate::parser::Parser;

        // Parse URL to extract file extension if present
        let url_path = url.split('?').next().unwrap_or(url);
        let extension = url_path
            .split('/')
            .next_back()
            .and_then(|filename| filename.rsplit('.').next());

        // Fetch data from URL using reqwest
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| OxirsError::Store(format!("Failed to create runtime: {e}")))?;

        let (content, content_type) = runtime.block_on(async {
            let response = reqwest::get(url)
                .await
                .map_err(|e| OxirsError::Store(format!("Failed to fetch URL {url}: {e}")))?;

            if !response.status().is_success() {
                return Err(OxirsError::Store(format!(
                    "HTTP error {} when fetching {url}",
                    response.status()
                )));
            }

            // Get content type from headers
            let content_type = response
                .headers()
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                .map(|s| s.split(';').next().unwrap_or(s).trim().to_string());

            // Get response body as text
            let text = response
                .text()
                .await
                .map_err(|e| OxirsError::Store(format!("Failed to read response body: {e}")))?;

            Ok::<_, OxirsError>((text, content_type))
        })?;

        // Detect RDF format from content type or file extension
        let format = Self::detect_format_from_url(&content_type, extension, &content)?;

        // Parse the content
        let parser = Parser::new(format);
        let quads = parser
            .parse_str_to_quads(&content)
            .map_err(|e| OxirsError::Store(format!("Failed to parse RDF data from {url}: {e}")))?;

        // Insert quads into the specified graph
        let target_graph = graph.cloned().map(GraphName::NamedNode);
        let mut inserted_count = 0;

        for quad in quads {
            // Override graph name if a target graph is specified
            let final_quad = if let Some(ref target) = target_graph {
                Quad::new(
                    quad.subject().clone(),
                    quad.predicate().clone(),
                    quad.object().clone(),
                    target.clone(),
                )
            } else {
                quad
            };

            if self.insert_quad(final_quad)? {
                inserted_count += 1;
            }
        }

        Ok(inserted_count)
    }

    /// Detect RDF format from content type, file extension, or content
    fn detect_format_from_url(
        content_type: &Option<String>,
        extension: Option<&str>,
        content: &str,
    ) -> Result<RdfFormat> {
        // Try to detect from content type first
        if let Some(ct) = content_type {
            let ct_lower = ct.to_lowercase();
            if let Some(format) = Self::format_from_media_type(&ct_lower) {
                return Ok(format);
            }
        }

        // Try to detect from file extension
        if let Some(ext) = extension {
            if let Some(format) = RdfFormat::from_extension(ext) {
                return Ok(format);
            }
        }

        // Try to detect from content
        if let Some(format) = crate::parser::detect_format_from_content(content) {
            return Ok(format);
        }

        // Default to N-Triples if we can't detect
        Err(OxirsError::Store(
            "Could not detect RDF format from URL, content type, or content".to_string(),
        ))
    }

    /// Map media type to RDF format
    fn format_from_media_type(media_type: &str) -> Option<RdfFormat> {
        match media_type {
            "text/turtle" | "application/x-turtle" => Some(RdfFormat::Turtle),
            "application/n-triples" | "text/plain" => Some(RdfFormat::NTriples),
            "application/trig" | "application/x-trig" => Some(RdfFormat::TriG),
            "application/n-quads" | "text/x-nquads" => Some(RdfFormat::NQuads),
            "application/rdf+xml" | "application/xml" | "text/xml" => Some(RdfFormat::RdfXml),
            "application/ld+json" | "application/json" => Some(RdfFormat::JsonLd),
            _ => None,
        }
    }

    /// Convert string to Subject term
    #[allow(dead_code)]
    fn string_to_subject(s: &str) -> Option<Subject> {
        if s.starts_with('<') && s.ends_with('>') {
            let iri = &s[1..s.len() - 1];
            NamedNode::new(iri).ok().map(Subject::NamedNode)
        } else if let Some(blank_id) = s.strip_prefix("_:") {
            BlankNode::new(blank_id).ok().map(Subject::BlankNode)
        } else {
            None
        }
    }

    /// Convert string to Predicate term
    #[allow(dead_code)]
    fn string_to_predicate(p: &str) -> Option<Predicate> {
        if p.starts_with('<') && p.ends_with('>') {
            let iri = &p[1..p.len() - 1];
            NamedNode::new(iri).ok().map(Predicate::NamedNode)
        } else {
            None
        }
    }

    /// Convert string to Object term
    #[allow(dead_code)]
    fn string_to_object(o: &str) -> Option<Object> {
        if o.starts_with('<') && o.ends_with('>') {
            let iri = &o[1..o.len() - 1];
            NamedNode::new(iri).ok().map(Object::NamedNode)
        } else if let Some(blank_id) = o.strip_prefix("_:") {
            BlankNode::new(blank_id).ok().map(Object::BlankNode)
        } else if o.starts_with('"') {
            // Simple literal parsing - more sophisticated parsing would be needed for real use
            let literal_content = &o[1..o.len() - 1];
            Some(Object::Literal(Literal::new(literal_content)))
        } else {
            None
        }
    }
}

impl Default for RdfStore {
    fn default() -> Self {
        RdfStore::new().unwrap()
    }
}

// Implement the Store trait for RdfStore using interior mutability
#[async_trait]
impl Store for RdfStore {
    fn insert_quad(&self, quad: Quad) -> Result<bool> {
        let inserted = match &self.backend {
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
                true // New quad inserted
            }
            StorageBackend::Memory(storage) => {
                let mut storage = storage.write().map_err(|e| {
                    crate::OxirsError::Store(format!("Failed to acquire write lock: {e}"))
                })?;
                storage.insert_quad(quad)
            }
            StorageBackend::Persistent(storage, _) => {
                let mut storage = storage.write().map_err(|e| {
                    crate::OxirsError::Store(format!("Failed to acquire write lock: {e}"))
                })?;
                let result = storage.insert_quad(quad);
                drop(storage); // Release lock before saving

                // Save to disk after insertion
                if result {
                    self.save_to_disk()?;
                }
                result
            }
        };

        Ok(inserted)
    }

    fn remove_quad(&self, quad: &Quad) -> Result<bool> {
        match &self.backend {
            StorageBackend::UltraMemory(index, _arena) => Ok(index.remove_quad(quad)),
            StorageBackend::Memory(storage) | StorageBackend::Persistent(storage, _) => {
                let mut storage = storage.write().map_err(|e| {
                    crate::OxirsError::Store(format!("Failed to acquire write lock: {e}"))
                })?;
                Ok(storage.remove_quad(quad))
            }
        }
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
