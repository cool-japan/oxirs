//! Oxigraph compatibility layer
//!
//! This module provides a compatibility layer that matches Oxigraph's API,
//! allowing OxiRS to be used as a drop-in replacement for Oxigraph.

use crate::{
    model::*,
    parser::RdfFormat,
    rdf_store::{OxirsQueryResults, RdfStore},
    transaction::{IsolationLevel, TransactionManager},
    OxirsError, Result, Store as OxirsStoreTrait,
};
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

/// Oxigraph-compatible store implementation
///
/// This provides the same API as oxigraph::Store for compatibility
///
/// Uses interior mutability to match Oxigraph's API where mutations take &self
pub struct Store {
    inner: Arc<RwLock<RdfStore>>,
    tx_manager: Arc<RwLock<Option<TransactionManager>>>,
    wal_dir: Option<PathBuf>,
}

impl Store {
    /// Creates a new in-memory store
    ///
    /// This matches oxigraph::Store::new()
    pub fn new() -> Result<Self> {
        Ok(Store {
            inner: Arc::new(RwLock::new(RdfStore::new()?)),
            tx_manager: Arc::new(RwLock::new(None)),
            wal_dir: None,
        })
    }

    /// Opens a persistent store at the given path
    ///
    /// This matches oxigraph::Store::open()
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_buf = path.as_ref().to_path_buf();
        let wal_dir = path_buf.join("wal");

        Ok(Store {
            inner: Arc::new(RwLock::new(RdfStore::open(&path_buf)?)),
            tx_manager: Arc::new(RwLock::new(None)),
            wal_dir: Some(wal_dir),
        })
    }

    /// Inserts a quad into the store
    ///
    /// Returns true if the quad was not already present
    pub fn insert<'a>(&self, quad: impl Into<QuadRef<'a>>) -> Result<bool> {
        let quad_ref = quad.into();
        let quad = Quad::new(
            quad_ref.subject().to_owned(),
            quad_ref.predicate().to_owned(),
            quad_ref.object().to_owned(),
            quad_ref.graph_name().to_owned(),
        );

        let store = self
            .inner
            .write()
            .map_err(|e| OxirsError::Store(format!("Failed to acquire write lock: {e}")))?;
        store.insert_quad(quad)
    }

    /// Extends the store with an iterator of quads
    pub fn extend<'a>(
        &self,
        quads: impl IntoIterator<Item = impl Into<QuadRef<'a>>>,
    ) -> Result<()> {
        let store = self
            .inner
            .write()
            .map_err(|e| OxirsError::Store(format!("Failed to acquire write lock: {e}")))?;

        for quad in quads {
            let quad_ref = quad.into();
            let quad = Quad::new(
                quad_ref.subject().to_owned(),
                quad_ref.predicate().to_owned(),
                quad_ref.object().to_owned(),
                quad_ref.graph_name().to_owned(),
            );
            store.insert_quad(quad)?;
        }

        Ok(())
    }

    /// Removes a quad from the store
    ///
    /// Returns true if the quad was present
    pub fn remove<'a>(&self, quad: impl Into<QuadRef<'a>>) -> Result<bool> {
        let quad_ref = quad.into();
        let quad = Quad::new(
            quad_ref.subject().to_owned(),
            quad_ref.predicate().to_owned(),
            quad_ref.object().to_owned(),
            quad_ref.graph_name().to_owned(),
        );

        let store = self
            .inner
            .write()
            .map_err(|e| OxirsError::Store(format!("Failed to acquire write lock: {e}")))?;
        store.remove_quad(&quad)
    }

    /// Loads a file into the store
    pub fn load_from_reader<R: BufRead>(
        &self,
        reader: R,
        format: RdfFormat,
        base_iri: Option<&str>,
        graph: Option<impl Into<GraphName>>,
    ) -> Result<()> {
        use crate::parser::Parser;

        // Read all data into a string
        let mut data = String::new();
        let mut reader = reader;
        // BufRead already extends Read, so this import is not needed
        reader
            .read_to_string(&mut data)
            .map_err(|e| OxirsError::Parse(format!("Failed to read input: {e}")))?;

        // Create parser with base IRI if provided
        let mut parser = Parser::new(format);
        if let Some(base) = base_iri {
            parser = parser.with_base_iri(base);
        }

        // Parse to quads
        let quads = parser.parse_str_to_quads(&data)?;

        // Get write lock on store
        let store = self
            .inner
            .write()
            .map_err(|e| OxirsError::Store(format!("Failed to acquire write lock: {e}")))?;

        // Insert quads, potentially modifying graph name
        let target_graph = graph.map(|g| g.into());
        for quad in quads {
            let final_quad = if let Some(ref g) = target_graph {
                // Override the quad's graph with the specified one
                Quad::new(
                    quad.subject().clone(),
                    quad.predicate().clone(),
                    quad.object().clone(),
                    g.clone(),
                )
            } else {
                quad
            };
            store.insert_quad(final_quad)?;
        }

        Ok(())
    }

    /// Dumps the store content to a writer
    pub fn dump_to_writer<'a, W: Write>(
        &self,
        mut writer: W,
        format: RdfFormat,
        graph: Option<impl Into<GraphNameRef<'a>>>,
    ) -> Result<()> {
        use crate::model::{dataset::Dataset, graph::Graph};
        use crate::serializer::Serializer;

        let store = self
            .inner
            .read()
            .map_err(|e| OxirsError::Store(format!("Failed to acquire read lock: {e}")))?;

        let serializer = Serializer::new(format);

        // Get quads to serialize
        let quads = if let Some(g) = graph {
            let graph_ref = g.into();
            let graph_name = graph_ref.to_owned();
            store.query_quads(None, None, None, Some(&graph_name))?
        } else {
            store.iter_quads()?
        };

        // Serialize based on format capabilities
        let output = match format {
            RdfFormat::Turtle | RdfFormat::NTriples | RdfFormat::RdfXml => {
                // These formats only support triples, so filter to default graph
                let triples: Vec<_> = quads
                    .into_iter()
                    .filter(|q| q.is_default_graph())
                    .map(|q| q.to_triple())
                    .collect();
                let graph = Graph::from_iter(triples);
                serializer.serialize_graph(&graph)?
            }
            RdfFormat::TriG | RdfFormat::NQuads | RdfFormat::JsonLd => {
                // These formats support quads/datasets
                let dataset = Dataset::from_iter(quads);
                serializer.serialize_dataset(&dataset)?
            }
        };

        writer
            .write_all(output.as_bytes())
            .map_err(|e| OxirsError::Serialize(format!("Failed to write output: {e}")))?;

        Ok(())
    }

    /// Checks if the store contains a given quad
    pub fn contains<'a>(&self, quad: impl Into<QuadRef<'a>>) -> Result<bool> {
        let quad_ref = quad.into();
        let quad = Quad::new(
            quad_ref.subject().to_owned(),
            quad_ref.predicate().to_owned(),
            quad_ref.object().to_owned(),
            quad_ref.graph_name().to_owned(),
        );

        let store = self
            .inner
            .read()
            .map_err(|e| OxirsError::Store(format!("Failed to acquire read lock: {e}")))?;
        store.contains_quad(&quad)
    }

    /// Returns the number of quads in the store
    pub fn len(&self) -> Result<usize> {
        let store = self
            .inner
            .read()
            .map_err(|e| OxirsError::Store(format!("Failed to acquire read lock: {e}")))?;
        store.len()
    }

    /// Checks if the store is empty
    pub fn is_empty(&self) -> Result<bool> {
        let store = self
            .inner
            .read()
            .map_err(|e| OxirsError::Store(format!("Failed to acquire read lock: {e}")))?;
        store.is_empty()
    }

    /// Returns an iterator over all quads matching a pattern
    pub fn quads_for_pattern<'a>(
        &self,
        subject: Option<impl Into<SubjectRef<'a>>>,
        predicate: Option<impl Into<PredicateRef<'a>>>,
        object: Option<impl Into<ObjectRef<'a>>>,
        graph_name: Option<impl Into<GraphNameRef<'a>>>,
    ) -> QuadIter {
        let subject = subject.map(|s| {
            let s_ref = s.into();
            match s_ref {
                SubjectRef::NamedNode(n) => Subject::NamedNode(n.to_owned()),
                SubjectRef::BlankNode(b) => Subject::BlankNode(b.to_owned()),
                SubjectRef::Variable(v) => Subject::Variable(v.to_owned()),
            }
        });

        let predicate = predicate.map(|p| {
            let p_ref = p.into();
            match p_ref {
                PredicateRef::NamedNode(n) => Predicate::NamedNode(n.to_owned()),
                PredicateRef::Variable(v) => Predicate::Variable(v.to_owned()),
            }
        });

        let object = object.map(|o| {
            let o_ref = o.into();
            match o_ref {
                ObjectRef::NamedNode(n) => Object::NamedNode(n.to_owned()),
                ObjectRef::BlankNode(b) => Object::BlankNode(b.to_owned()),
                ObjectRef::Literal(l) => Object::Literal(l.to_owned()),
                ObjectRef::Variable(v) => Object::Variable(v.to_owned()),
            }
        });

        let graph_name = graph_name.map(|g| {
            let g_ref = g.into();
            g_ref.to_owned()
        });

        // Query the inner store
        let quads = match self.inner.read() {
            Ok(store) => store
                .query_quads(
                    subject.as_ref(),
                    predicate.as_ref(),
                    object.as_ref(),
                    graph_name.as_ref(),
                )
                .unwrap_or_default(),
            _ => Vec::new(),
        };

        QuadIter { quads, index: 0 }
    }

    /// Returns an iterator over all quads in the store
    pub fn iter(&self) -> QuadIter {
        self.quads_for_pattern(
            None::<SubjectRef>,
            None::<PredicateRef>,
            None::<ObjectRef>,
            None::<GraphNameRef>,
        )
    }

    /// Returns all named graphs in the store
    pub fn named_graphs(&self) -> GraphNameIter {
        // Collect unique graph names from all quads
        let mut graph_names = std::collections::HashSet::new();
        if let Ok(store) = self.inner.read() {
            if let Ok(quads) = store.iter_quads() {
                for quad in quads {
                    if let GraphName::NamedNode(n) = quad.graph_name() {
                        graph_names.insert(n.clone());
                    }
                }
            }
        }

        GraphNameIter {
            graphs: graph_names.into_iter().collect(),
            index: 0,
        }
    }

    /// Checks if the store contains a given named graph
    pub fn contains_named_graph<'a>(
        &self,
        graph_name: impl Into<NamedOrBlankNodeRef<'a>>,
    ) -> Result<bool> {
        let graph_ref = graph_name.into();
        let graph = match graph_ref {
            NamedOrBlankNodeRef::NamedNode(n) => GraphName::NamedNode(n.to_owned()),
            NamedOrBlankNodeRef::BlankNode(b) => GraphName::BlankNode(b.to_owned()),
        };

        // Check if any quads exist in this graph
        let store = self
            .inner
            .read()
            .map_err(|e| OxirsError::Store(format!("Failed to acquire read lock: {e}")))?;
        let quads = store.query_quads(None, None, None, Some(&graph))?;
        Ok(!quads.is_empty())
    }

    /// Clears the store
    pub fn clear(&self) -> Result<()> {
        let mut store = self
            .inner
            .write()
            .map_err(|e| OxirsError::Store(format!("Failed to acquire write lock: {e}")))?;
        store.clear()
    }

    /// Clears a specific graph
    pub fn clear_graph<'a>(&self, graph_name: impl Into<GraphNameRef<'a>>) -> Result<()> {
        let graph_ref = graph_name.into();
        let graph = graph_ref.to_owned();

        let store = self
            .inner
            .write()
            .map_err(|e| OxirsError::Store(format!("Failed to acquire write lock: {e}")))?;

        // Get all quads in the specified graph
        let quads_to_remove = store.query_quads(None, None, None, Some(&graph))?;

        // Remove each quad
        for quad in quads_to_remove {
            store.remove_quad(&quad)?;
        }

        Ok(())
    }

    /// Executes a SPARQL query
    pub fn query(&self, query: &str) -> Result<QueryResults> {
        let store = self
            .inner
            .read()
            .map_err(|e| OxirsError::Store(format!("Failed to acquire read lock: {e}")))?;
        let results = store.query(query)?;
        Ok(QueryResults { inner: results })
    }

    /// Executes a SPARQL update
    pub fn update(&self, update_str: &str) -> Result<()> {
        use crate::query::{UpdateExecutor, UpdateParser};

        // Parse the UPDATE string
        let parser = UpdateParser::new();
        let update = parser.parse(update_str)?;

        // Get write access to the store
        let store = self
            .inner
            .write()
            .map_err(|e| OxirsError::Store(format!("Failed to acquire write lock: {e}")))?;

        // Execute the update
        let executor = UpdateExecutor::new(&*store);
        executor.execute(&update)?;

        Ok(())
    }

    /// Creates a transaction for the store
    ///
    /// This method provides ACID transaction support with automatic commit/abort handling.
    /// The transaction uses Snapshot isolation level by default.
    ///
    /// # Example
    ///
    /// ```ignore
    /// store.transaction(|tx| {
    ///     // Perform transactional operations
    ///     Ok(())
    /// })?;
    /// ```
    pub fn transaction<T, E>(
        &self,
        f: impl FnOnce(&mut crate::AcidTransaction) -> std::result::Result<T, E>,
    ) -> std::result::Result<T, E>
    where
        E: From<OxirsError>,
    {
        // Ensure TransactionManager is initialized
        self.ensure_tx_manager()?;

        // Get the transaction manager
        let mut tx_mgr_guard = self
            .tx_manager
            .write()
            .map_err(|e| E::from(OxirsError::Store(format!("Failed to acquire lock: {e}"))))?;

        let tx_mgr = tx_mgr_guard.as_mut().ok_or_else(|| {
            E::from(OxirsError::Store(
                "Transaction manager not initialized".to_string(),
            ))
        })?;

        // Begin a transaction with Snapshot isolation
        let mut transaction = tx_mgr.begin(IsolationLevel::Snapshot).map_err(E::from)?;

        // Execute the user function
        let result = f(&mut transaction);

        // Commit the transaction if the function succeeded
        match result {
            Ok(value) => {
                transaction.commit().map_err(E::from)?;
                Ok(value)
            }
            Err(error) => {
                let _ = transaction.abort();
                Err(error)
            }
        }
    }

    /// Ensures the transaction manager is initialized
    fn ensure_tx_manager(&self) -> Result<()> {
        let mut tx_mgr_guard = self
            .tx_manager
            .write()
            .map_err(|e| OxirsError::Store(format!("Failed to acquire lock: {e}")))?;

        if tx_mgr_guard.is_none() {
            // Determine WAL directory
            let wal_dir = if let Some(ref wal_path) = self.wal_dir {
                wal_path.clone()
            } else {
                // Use temporary directory for in-memory stores
                std::env::temp_dir().join("oxirs_wal")
            };

            // Create the transaction manager
            let tx_mgr = TransactionManager::new(&wal_dir)?;
            *tx_mgr_guard = Some(tx_mgr);
        }

        Ok(())
    }

    /// Validates the store integrity
    pub fn validate(&self) -> Result<()> {
        // OxiRS doesn't have explicit validation yet
        Ok(())
    }

    /// Optimizes the store layout
    pub fn optimize(&self) -> Result<()> {
        // Trigger arena cleanup if using ultra-performance mode
        let store = self
            .inner
            .read()
            .map_err(|e| OxirsError::Store(format!("Failed to acquire read lock: {e}")))?;
        store.clear_arena();
        Ok(())
    }

    /// Backs up the store to a path
    pub fn backup<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        use crate::parser::RdfFormat;
        use crate::serializer::Serializer;
        use std::fs::File;
        use std::io::Write;
        use std::time::{SystemTime, UNIX_EPOCH};

        let backup_path = path.as_ref();

        // Create backup directory if it doesn't exist
        if let Some(parent) = backup_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                OxirsError::Store(format!("Failed to create backup directory: {e}"))
            })?;
        }

        // Generate backup filename with timestamp
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let backup_file_path = if backup_path.is_dir() {
            backup_path.join(format!("oxirs_backup_{timestamp}.nq"))
        } else {
            backup_path.to_path_buf()
        };

        // Get read lock on the store
        let store = self
            .inner
            .read()
            .map_err(|e| OxirsError::Store(format!("Failed to acquire read lock: {e}")))?;

        // Get all quads from the store
        let quads = store
            .iter_quads()
            .map_err(|e| OxirsError::Store(format!("Failed to iterate quads: {e}")))?;

        // Create dataset from quads
        let dataset = crate::model::dataset::Dataset::from_iter(quads.clone());

        // Serialize to N-Quads format (most portable and complete format)
        let serializer = Serializer::new(RdfFormat::NQuads);
        let serialized_data = serializer
            .serialize_dataset(&dataset)
            .map_err(|e| OxirsError::Store(format!("Failed to serialize dataset: {e}")))?;

        // Write to backup file
        let mut backup_file = File::create(&backup_file_path)
            .map_err(|e| OxirsError::Store(format!("Failed to create backup file: {e}")))?;

        backup_file
            .write_all(serialized_data.as_bytes())
            .map_err(|e| OxirsError::Store(format!("Failed to write backup data: {e}")))?;

        backup_file
            .sync_all()
            .map_err(|e| OxirsError::Store(format!("Failed to sync backup file: {e}")))?;

        // Calculate backup size for logging
        let backup_size = serialized_data.len();
        let quad_count = quads.len();

        tracing::info!(
            "Store backup completed successfully. File: {}, Quads: {}, Size: {} bytes",
            backup_file_path.display(),
            quad_count,
            backup_size
        );

        Ok(())
    }

    /// Flushes any pending changes to disk
    pub fn flush(&self) -> Result<()> {
        // No-op for now as OxiRS doesn't buffer writes
        Ok(())
    }
}

impl Default for Store {
    fn default() -> Self {
        Store::new().expect("Store::new() should not fail")
    }
}

/// Iterator over quads (Oxigraph-compatible)
pub struct QuadIter {
    quads: Vec<Quad>,
    index: usize,
}

impl Iterator for QuadIter {
    type Item = Quad;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.quads.len() {
            let quad = self.quads[self.index].clone();
            self.index += 1;
            Some(quad)
        } else {
            None
        }
    }
}

/// Iterator over graph names (Oxigraph-compatible)
pub struct GraphNameIter {
    graphs: Vec<NamedNode>,
    index: usize,
}

impl Iterator for GraphNameIter {
    type Item = NamedNode;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.graphs.len() {
            let graph = self.graphs[self.index].clone();
            self.index += 1;
            Some(graph)
        } else {
            None
        }
    }
}

/// Oxigraph-compatible query results
pub struct QueryResults {
    #[allow(dead_code)]
    inner: OxirsQueryResults,
}

impl QueryResults {
    /// Returns true if the results are a boolean
    pub fn is_boolean(&self) -> bool {
        matches!(
            self.inner.results(),
            crate::rdf_store::types::QueryResults::Boolean(_)
        )
    }

    /// Returns the boolean value if the results are a boolean
    pub fn boolean(&self) -> Option<bool> {
        match self.inner.results() {
            crate::rdf_store::types::QueryResults::Boolean(b) => Some(*b),
            _ => None,
        }
    }

    /// Returns true if the results are solutions
    pub fn is_solutions(&self) -> bool {
        matches!(
            self.inner.results(),
            crate::rdf_store::types::QueryResults::Bindings(_)
        )
    }

    /// Returns true if the results are a graph
    pub fn is_graph(&self) -> bool {
        matches!(
            self.inner.results(),
            crate::rdf_store::types::QueryResults::Graph(_)
        )
    }
}

/// Oxigraph-compatible transaction
///
/// Note: This is a placeholder implementation. Full transactional support
/// would require implementing proper transaction isolation in OxiRS.
pub struct Transaction {
    // Placeholder for future transaction implementation
    operations: Vec<TransactionOp>,
}

enum TransactionOp {
    #[allow(dead_code)]
    Insert(Quad),
    #[allow(dead_code)]
    Remove(Quad),
}

impl Transaction {
    #[allow(dead_code)]
    fn new() -> Self {
        Transaction {
            operations: Vec::new(),
        }
    }

    /// Inserts a quad in the transaction
    pub fn insert<'b>(&mut self, quad: impl Into<QuadRef<'b>>) -> Result<bool> {
        let quad_ref = quad.into();
        let quad = Quad::new(
            quad_ref.subject().to_owned(),
            quad_ref.predicate().to_owned(),
            quad_ref.object().to_owned(),
            quad_ref.graph_name().to_owned(),
        );
        self.operations.push(TransactionOp::Insert(quad));
        Ok(true) // Optimistically return true
    }

    /// Removes a quad in the transaction
    pub fn remove<'b>(&mut self, quad: impl Into<QuadRef<'b>>) -> Result<bool> {
        let quad_ref = quad.into();
        let quad = Quad::new(
            quad_ref.subject().to_owned(),
            quad_ref.predicate().to_owned(),
            quad_ref.object().to_owned(),
            quad_ref.graph_name().to_owned(),
        );
        self.operations.push(TransactionOp::Remove(quad));
        Ok(true) // Optimistically return true
    }
}

/// Oxigraph-compatible error type
#[derive(Debug, thiserror::Error)]
pub enum OxigraphCompatError {
    #[error("Store error: {0}")]
    Store(String),
    #[error("Parse error: {0}")]
    Parse(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

impl From<OxirsError> for OxigraphCompatError {
    fn from(err: OxirsError) -> Self {
        match err {
            OxirsError::Store(msg) => OxigraphCompatError::Store(msg),
            OxirsError::Parse(msg) => OxigraphCompatError::Parse(msg),
            _ => OxigraphCompatError::Store(err.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Literal, NamedNode};
    use crate::parser::RdfFormat;
    use std::io::Cursor;

    #[test]
    fn test_oxigraph_compat_store_creation() {
        let store = Store::new().unwrap();
        assert!(store.is_empty().unwrap());
        assert_eq!(store.len().unwrap(), 0);
    }

    #[test]
    fn test_oxigraph_compat_insert_and_query() {
        let store = Store::new().unwrap();

        // Create test quad
        let subject = NamedNode::new("http://example.org/subject").unwrap();
        let predicate = NamedNode::new("http://example.org/predicate").unwrap();
        let object = Literal::new("test object");
        let graph = NamedNode::new("http://example.org/graph").unwrap();

        let quad = Quad::new(
            subject.clone(),
            predicate.clone(),
            object.clone(),
            graph.clone(),
        );

        // Insert quad
        assert!(store.insert(QuadRef::from(&quad)).unwrap());
        assert_eq!(store.len().unwrap(), 1);
        assert!(!store.is_empty().unwrap());

        // Check contains
        assert!(store.contains(QuadRef::from(&quad)).unwrap());

        // Query by pattern
        let quads: Vec<_> = store
            .quads_for_pattern(
                Some(SubjectRef::NamedNode(&subject)),
                None::<PredicateRef>,
                None::<ObjectRef>,
                None::<GraphNameRef>,
            )
            .collect();
        assert_eq!(quads.len(), 1);
        assert_eq!(quads[0], quad);

        // Remove quad
        assert!(store.remove(QuadRef::from(&quad)).unwrap());
        assert!(store.is_empty().unwrap());
    }

    #[test]
    fn test_oxigraph_compat_extend() {
        let store = Store::new().unwrap();

        let quads = [
            Quad::new(
                NamedNode::new("http://example.org/s1").unwrap(),
                NamedNode::new("http://example.org/p1").unwrap(),
                Literal::new("o1"),
                GraphName::DefaultGraph,
            ),
            Quad::new(
                NamedNode::new("http://example.org/s2").unwrap(),
                NamedNode::new("http://example.org/p2").unwrap(),
                Literal::new("o2"),
                NamedNode::new("http://example.org/g1").unwrap(),
            ),
        ];

        store.extend(quads.iter().map(QuadRef::from)).unwrap();
        assert_eq!(store.len().unwrap(), 2);
    }

    #[test]
    fn test_oxigraph_compat_named_graphs() {
        let store = Store::new().unwrap();

        // Create nodes
        let s1 = NamedNode::new("http://example.org/s1").unwrap();
        let s2 = NamedNode::new("http://example.org/s2").unwrap();
        let p1 = NamedNode::new("http://example.org/p1").unwrap();
        let p2 = NamedNode::new("http://example.org/p2").unwrap();
        let o1 = Literal::new("o1");
        let o2 = Literal::new("o2");
        let g1 = NamedNode::new("http://example.org/g1").unwrap();
        let g2 = NamedNode::new("http://example.org/g2").unwrap();

        // Insert quads in different graphs
        store
            .insert(QuadRef::new(
                SubjectRef::NamedNode(&s1),
                PredicateRef::NamedNode(&p1),
                ObjectRef::Literal(&o1),
                GraphNameRef::NamedNode(&g1),
            ))
            .unwrap();

        store
            .insert(QuadRef::new(
                SubjectRef::NamedNode(&s2),
                PredicateRef::NamedNode(&p2),
                ObjectRef::Literal(&o2),
                GraphNameRef::NamedNode(&g2),
            ))
            .unwrap();

        // Check named graphs
        let graphs: Vec<_> = store.named_graphs().collect();
        assert_eq!(graphs.len(), 2);
        assert!(graphs.contains(&g1));
        assert!(graphs.contains(&g2));

        // Check contains_named_graph
        assert!(store
            .contains_named_graph(NamedOrBlankNodeRef::NamedNode(&g1))
            .unwrap());
        assert!(store
            .contains_named_graph(NamedOrBlankNodeRef::NamedNode(&g2))
            .unwrap());
    }

    #[test]
    fn test_oxigraph_compat_clear_graph() {
        let store = Store::new().unwrap();

        // Create nodes
        let s1 = NamedNode::new("http://example.org/s1").unwrap();
        let s2 = NamedNode::new("http://example.org/s2").unwrap();
        let p1 = NamedNode::new("http://example.org/p1").unwrap();
        let p2 = NamedNode::new("http://example.org/p2").unwrap();
        let o1 = Literal::new("o1");
        let o2 = Literal::new("o2");
        let graph = NamedNode::new("http://example.org/graph").unwrap();

        // Add quads to specific graph and default graph
        store
            .insert(QuadRef::new(
                SubjectRef::NamedNode(&s1),
                PredicateRef::NamedNode(&p1),
                ObjectRef::Literal(&o1),
                GraphNameRef::NamedNode(&graph),
            ))
            .unwrap();

        store
            .insert(QuadRef::new(
                SubjectRef::NamedNode(&s2),
                PredicateRef::NamedNode(&p2),
                ObjectRef::Literal(&o2),
                GraphNameRef::DefaultGraph,
            ))
            .unwrap();

        assert_eq!(store.len().unwrap(), 2);

        // Clear specific graph
        store.clear_graph(GraphNameRef::NamedNode(&graph)).unwrap();
        assert_eq!(store.len().unwrap(), 1); // Only default graph quad remains

        // Clear all
        store.clear().unwrap();
        assert!(store.is_empty().unwrap());
    }

    #[test]
    fn test_oxigraph_compat_load_from_reader() {
        let store = Store::new().unwrap();

        let turtle_data = r#"
            @prefix ex: <http://example.org/> .
            ex:subject ex:predicate "object" .
        "#;

        let reader = Cursor::new(turtle_data.as_bytes());
        store
            .load_from_reader(
                reader,
                RdfFormat::Turtle,
                Some("http://example.org/"),
                None::<GraphName>,
            )
            .unwrap();

        assert_eq!(store.len().unwrap(), 1);

        // Verify the loaded data
        let quads: Vec<_> = store.iter().collect();
        assert_eq!(quads.len(), 1);
        assert_eq!(
            quads[0].subject().to_string(),
            "<http://example.org/subject>"
        );
        assert_eq!(
            quads[0].predicate().to_string(),
            "<http://example.org/predicate>"
        );
    }

    #[test]
    fn test_oxigraph_compat_dump_to_writer() {
        let store = Store::new().unwrap();

        // Create nodes
        let subject = NamedNode::new("http://example.org/subject").unwrap();
        let predicate = NamedNode::new("http://example.org/predicate").unwrap();
        let object = Literal::new("object");

        // Add some test data
        store
            .insert(QuadRef::new(
                SubjectRef::NamedNode(&subject),
                PredicateRef::NamedNode(&predicate),
                ObjectRef::Literal(&object),
                GraphNameRef::DefaultGraph,
            ))
            .unwrap();

        // Dump to N-Triples format
        let mut output = Vec::new();
        store
            .dump_to_writer(&mut output, RdfFormat::NTriples, None::<GraphNameRef>)
            .unwrap();

        let result = String::from_utf8(output).unwrap();
        assert!(result.contains("<http://example.org/subject>"));
        assert!(result.contains("<http://example.org/predicate>"));
        assert!(result.contains("\"object\""));
    }
}
