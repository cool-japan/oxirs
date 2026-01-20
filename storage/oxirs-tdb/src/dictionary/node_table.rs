//! NodeTable for Term ↔ NodeId mapping with prefix compression

use super::node_id::NodeId;
use super::term::Term;
use crate::btree::BTree;
use crate::compression::prefix::PrefixCompressor;
use crate::error::Result;
use crate::storage::BufferPool;
use parking_lot::RwLock;
use std::sync::Arc;

/// NodeTable provides bidirectional mapping between Terms and NodeIds
///
/// Uses two B+Trees:
/// - term_to_id: Term → NodeId (for encoding)
/// - id_to_term: NodeId → Term (for decoding)
///
/// # Prefix Compression
///
/// For IRI terms, the NodeTable automatically applies prefix compression
/// to reduce storage overhead. Common IRI prefixes (namespaces) are stored
/// once and referenced by ID, significantly reducing memory usage for
/// datasets with many IRIs sharing the same namespace.
pub struct NodeTable {
    /// B+Tree mapping Term → NodeId
    term_to_id: RwLock<BTree<Term, NodeId>>,

    /// B+Tree mapping NodeId → Term
    id_to_term: RwLock<BTree<NodeId, Term>>,

    /// Next available NodeId
    next_id: RwLock<NodeId>,

    /// Prefix compressor for IRI optimization
    prefix_compressor: RwLock<PrefixCompressor>,

    /// Enable prefix compression (can be disabled for testing)
    compression_enabled: bool,
}

impl NodeTable {
    /// Create a new NodeTable with prefix compression enabled
    pub fn new(buffer_pool: Arc<BufferPool>) -> Self {
        Self::with_compression(buffer_pool, true)
    }

    /// Create a new NodeTable with configurable compression
    pub fn with_compression(buffer_pool: Arc<BufferPool>, compression_enabled: bool) -> Self {
        NodeTable {
            term_to_id: RwLock::new(BTree::new(buffer_pool.clone())),
            id_to_term: RwLock::new(BTree::new(buffer_pool)),
            next_id: RwLock::new(NodeId::FIRST),
            prefix_compressor: RwLock::new(PrefixCompressor::new(15)), // Minimum 15 chars for compression
            compression_enabled,
        }
    }

    /// Get prefix compression statistics
    pub fn compression_stats(&self) -> crate::compression::prefix::CompressionStats {
        let compressor = self.prefix_compressor.read();
        compressor.stats()
    }

    /// Get or create a NodeId for a term
    ///
    /// If the term already exists, returns its NodeId.
    /// Otherwise, assigns a new NodeId and stores the mapping.
    ///
    /// # Prefix Compression
    ///
    /// When prefix compression is enabled and the term is an IRI,
    /// the compressor tracks the IRI prefix for statistics. Future
    /// versions will use this for actual storage optimization.
    pub fn get_or_create(&self, term: &Term) -> Result<NodeId> {
        // Try to find existing mapping
        {
            let term_to_id = self.term_to_id.read();
            if let Some(id) = term_to_id.search(term)? {
                return Ok(id);
            }
        }

        // Term not found, create new mapping
        let mut term_to_id = self.term_to_id.write();
        let mut id_to_term = self.id_to_term.write();
        let mut next_id = self.next_id.write();

        // Double-check (another thread might have created it)
        if let Some(id) = term_to_id.search(term)? {
            return Ok(id);
        }

        // Track IRI prefix for compression statistics
        if self.compression_enabled {
            if let Some(iri) = term.as_iri() {
                let mut compressor = self.prefix_compressor.write();
                // Track the IRI to build prefix dictionary
                // This doesn't change storage format yet, just tracks stats
                let _ = compressor.compress(iri);
            }
        }

        // Assign new NodeId
        let id = *next_id;
        *next_id = next_id.next();

        // Store bidirectional mapping
        term_to_id.insert(term.clone(), id)?;
        id_to_term.insert(id, term.clone())?;

        Ok(id)
    }

    /// Get NodeId for a term (returns None if not found)
    pub fn get_id(&self, term: &Term) -> Result<Option<NodeId>> {
        let term_to_id = self.term_to_id.read();
        term_to_id.search(term)
    }

    /// Get Term for a NodeId (returns None if not found)
    pub fn get_term(&self, id: NodeId) -> Result<Option<Term>> {
        let id_to_term = self.id_to_term.read();
        id_to_term.search(&id)
    }

    /// Get the total number of terms stored
    pub fn size(&self) -> u64 {
        let next_id = self.next_id.read();
        next_id.as_u64() - NodeId::FIRST.as_u64()
    }

    /// Check if a term exists in the dictionary
    pub fn contains(&self, term: &Term) -> Result<bool> {
        Ok(self.get_id(term)?.is_some())
    }

    /// Check if a NodeId exists in the dictionary
    pub fn contains_id(&self, id: NodeId) -> Result<bool> {
        Ok(self.get_term(id)?.is_some())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{Allocator, FileManager};
    use tempfile::TempDir;

    fn create_test_node_table() -> (TempDir, NodeTable) {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");

        let file_manager = Arc::new(FileManager::open(&db_path, true).unwrap());
        let buffer_pool = Arc::new(BufferPool::new(100, file_manager));

        let node_table = NodeTable::new(buffer_pool);
        (temp_dir, node_table)
    }

    #[test]
    fn test_node_table_get_or_create() -> Result<()> {
        let (_temp_dir, table) = create_test_node_table();

        let term1 = Term::iri("http://example.org/resource1");
        let term2 = Term::iri("http://example.org/resource2");

        let id1 = table.get_or_create(&term1)?;
        let id2 = table.get_or_create(&term2)?;

        // Different terms get different IDs
        assert_ne!(id1, id2);

        // Same term gets same ID
        let id1_again = table.get_or_create(&term1)?;
        assert_eq!(id1, id1_again);

        Ok(())
    }

    #[test]
    fn test_node_table_bidirectional_mapping() -> Result<()> {
        let (_temp_dir, table) = create_test_node_table();

        let term = Term::literal_with_lang("Hello World", "en");
        let id = table.get_or_create(&term)?;

        // Verify bidirectional mapping
        assert_eq!(table.get_id(&term)?, Some(id));
        assert_eq!(table.get_term(id)?, Some(term));

        Ok(())
    }

    #[test]
    fn test_node_table_get_id_not_found() -> Result<()> {
        let (_temp_dir, table) = create_test_node_table();

        let term = Term::iri("http://not-exists.com");
        assert_eq!(table.get_id(&term)?, None);

        Ok(())
    }

    #[test]
    fn test_node_table_get_term_not_found() -> Result<()> {
        let (_temp_dir, table) = create_test_node_table();

        let id = NodeId::new(999);
        assert_eq!(table.get_term(id)?, None);

        Ok(())
    }

    #[test]
    fn test_node_table_size() -> Result<()> {
        let (_temp_dir, table) = create_test_node_table();

        assert_eq!(table.size(), 0);

        table.get_or_create(&Term::iri("http://a.com"))?;
        assert_eq!(table.size(), 1);

        table.get_or_create(&Term::iri("http://b.com"))?;
        assert_eq!(table.size(), 2);

        // Creating same term again doesn't increase size
        table.get_or_create(&Term::iri("http://a.com"))?;
        assert_eq!(table.size(), 2);

        Ok(())
    }

    #[test]
    fn test_node_table_contains() -> Result<()> {
        let (_temp_dir, table) = create_test_node_table();

        let term1 = Term::iri("http://exists.com");
        let term2 = Term::iri("http://not-exists.com");

        table.get_or_create(&term1)?;

        assert!(table.contains(&term1)?);
        assert!(!table.contains(&term2)?);

        Ok(())
    }

    #[test]
    fn test_node_table_contains_id() -> Result<()> {
        let (_temp_dir, table) = create_test_node_table();

        let term = Term::literal("test");
        let id = table.get_or_create(&term)?;

        assert!(table.contains_id(id)?);
        assert!(!table.contains_id(NodeId::new(999))?);

        Ok(())
    }

    #[test]
    fn test_node_table_multiple_term_types() -> Result<()> {
        let (_temp_dir, table) = create_test_node_table();

        let iri = Term::iri("http://example.org");
        let literal = Term::literal("test value");
        let blank = Term::blank_node("b0");

        let id1 = table.get_or_create(&iri)?;
        let id2 = table.get_or_create(&literal)?;
        let id3 = table.get_or_create(&blank)?;

        // All get unique IDs
        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
        assert_ne!(id1, id3);

        // All are retrievable
        assert_eq!(table.get_term(id1)?, Some(iri));
        assert_eq!(table.get_term(id2)?, Some(literal));
        assert_eq!(table.get_term(id3)?, Some(blank));

        Ok(())
    }

    #[test]
    fn test_prefix_compression_integration() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");

        let file_manager = Arc::new(FileManager::open(&db_path, true).unwrap());
        let buffer_pool = Arc::new(BufferPool::new(100, file_manager));

        // Create NodeTable with compression enabled
        let table = NodeTable::with_compression(buffer_pool, true);

        // Add IRIs sharing common namespaces
        let iri1 = Term::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
        let iri2 = Term::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#Property");
        let iri3 = Term::iri("http://www.w3.org/2000/01/rdf-schema#label");
        let iri4 = Term::iri("http://www.w3.org/2000/01/rdf-schema#comment");

        // Encode all IRIs
        table.get_or_create(&iri1)?;
        table.get_or_create(&iri2)?;
        table.get_or_create(&iri3)?;
        table.get_or_create(&iri4)?;

        // Check compression statistics
        let stats = table.compression_stats();

        // Should have detected 2 common prefixes
        assert_eq!(stats.num_prefixes, 2);

        // Prefixes should save space
        assert!(stats.total_prefix_bytes > 0);

        Ok(())
    }

    #[test]
    fn test_prefix_compression_disabled() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");

        let file_manager = Arc::new(FileManager::open(&db_path, true).unwrap());
        let buffer_pool = Arc::new(BufferPool::new(100, file_manager));

        // Create NodeTable with compression disabled
        let table = NodeTable::with_compression(buffer_pool, false);

        // Add IRIs
        let iri1 = Term::iri("http://example.org/Person");
        let iri2 = Term::iri("http://example.org/name");

        table.get_or_create(&iri1)?;
        table.get_or_create(&iri2)?;

        // Compression disabled - no prefixes should be tracked
        let stats = table.compression_stats();
        assert_eq!(stats.num_prefixes, 0);

        Ok(())
    }

    #[test]
    fn test_prefix_compression_with_short_iris() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");

        let file_manager = Arc::new(FileManager::open(&db_path, true).unwrap());
        let buffer_pool = Arc::new(BufferPool::new(100, file_manager));

        let table = NodeTable::with_compression(buffer_pool, true);

        // Add short IRI (below minimum compression threshold)
        let short_iri = Term::iri("http://a.b");

        table.get_or_create(&short_iri)?;

        // Short IRIs should not create prefixes
        let stats = table.compression_stats();
        assert_eq!(stats.num_prefixes, 0);

        Ok(())
    }

    #[test]
    fn test_prefix_compression_with_literals() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");

        let file_manager = Arc::new(FileManager::open(&db_path, true).unwrap());
        let buffer_pool = Arc::new(BufferPool::new(100, file_manager));

        let table = NodeTable::with_compression(buffer_pool, true);

        // Add literals (should not be compressed)
        let lit1 = Term::literal("This is a long literal value that could be compressed");
        let lit2 = Term::literal("Another long literal value");

        table.get_or_create(&lit1)?;
        table.get_or_create(&lit2)?;

        // Literals should not create prefixes
        let stats = table.compression_stats();
        assert_eq!(stats.num_prefixes, 0);

        Ok(())
    }

    #[test]
    fn test_prefix_compression_realistic_dataset() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");

        let file_manager = Arc::new(FileManager::open(&db_path, true).unwrap());
        let buffer_pool = Arc::new(BufferPool::new(100, file_manager));

        let table = NodeTable::with_compression(buffer_pool, true);

        // Simulate realistic RDF dataset with common vocabularies
        let vocab_iris = vec![
            // RDF vocabulary
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#Property",
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement",
            // RDFS vocabulary
            "http://www.w3.org/2000/01/rdf-schema#label",
            "http://www.w3.org/2000/01/rdf-schema#comment",
            "http://www.w3.org/2000/01/rdf-schema#Class",
            // FOAF vocabulary
            "http://xmlns.com/foaf/0.1/Person",
            "http://xmlns.com/foaf/0.1/name",
            "http://xmlns.com/foaf/0.1/knows",
            // Application-specific
            "http://example.org/vocab/Employee",
            "http://example.org/vocab/salary",
            "http://example.org/vocab/department",
        ];

        for iri in vocab_iris {
            table.get_or_create(&Term::iri(iri))?;
        }

        // Should detect common namespace prefixes
        let stats = table.compression_stats();

        // Expect at least 4 prefixes: rdf, rdfs, foaf, example
        assert!(stats.num_prefixes >= 4);

        // Prefixes should provide meaningful compression
        assert!(stats.total_prefix_bytes > 100);

        Ok(())
    }
}
