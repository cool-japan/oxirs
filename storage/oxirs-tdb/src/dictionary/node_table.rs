//! NodeTable for Term ↔ NodeId mapping

use super::node_id::NodeId;
use super::term::Term;
use crate::btree::BTree;
use crate::error::Result;
use crate::storage::BufferPool;
use parking_lot::RwLock;
use std::sync::Arc;

/// NodeTable provides bidirectional mapping between Terms and NodeIds
///
/// Uses two B+Trees:
/// - term_to_id: Term → NodeId (for encoding)
/// - id_to_term: NodeId → Term (for decoding)
pub struct NodeTable {
    /// B+Tree mapping Term → NodeId
    term_to_id: RwLock<BTree<Term, NodeId>>,

    /// B+Tree mapping NodeId → Term
    id_to_term: RwLock<BTree<NodeId, Term>>,

    /// Next available NodeId
    next_id: RwLock<NodeId>,
}

impl NodeTable {
    /// Create a new NodeTable
    pub fn new(buffer_pool: Arc<BufferPool>) -> Self {
        NodeTable {
            term_to_id: RwLock::new(BTree::new(buffer_pool.clone())),
            id_to_term: RwLock::new(BTree::new(buffer_pool)),
            next_id: RwLock::new(NodeId::FIRST),
        }
    }

    /// Get or create a NodeId for a term
    ///
    /// If the term already exists, returns its NodeId.
    /// Otherwise, assigns a new NodeId and stores the mapping.
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
}
