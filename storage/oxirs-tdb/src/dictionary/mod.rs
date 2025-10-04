//! Dictionary encoding for IRIs and Literals
//!
//! This module implements dictionary encoding to compress
//! IRIs and literals to 8-byte NodeIDs.

pub mod node_id;
pub mod term;
pub mod node_table;

pub use node_id::NodeId;
pub use term::Term;
pub use node_table::NodeTable;

use crate::error::Result;
use crate::storage::BufferPool;
use std::sync::Arc;

/// High-level dictionary interface for RDF term encoding
pub struct Dictionary {
    node_table: NodeTable,
}

impl Dictionary {
    /// Create a new dictionary
    pub fn new(buffer_pool: Arc<BufferPool>) -> Self {
        Dictionary {
            node_table: NodeTable::new(buffer_pool),
        }
    }

    /// Encode a term to a NodeId (creates new mapping if needed)
    pub fn encode(&self, term: &Term) -> Result<NodeId> {
        self.node_table.get_or_create(term)
    }

    /// Decode a NodeId back to a Term
    pub fn decode(&self, id: NodeId) -> Result<Option<Term>> {
        self.node_table.get_term(id)
    }

    /// Get NodeId for a term without creating it
    pub fn lookup(&self, term: &Term) -> Result<Option<NodeId>> {
        self.node_table.get_id(term)
    }

    /// Check if a term is in the dictionary
    pub fn contains(&self, term: &Term) -> Result<bool> {
        self.node_table.contains(term)
    }

    /// Get the total number of terms in the dictionary
    pub fn size(&self) -> u64 {
        self.node_table.size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::FileManager;
    use tempfile::TempDir;

    fn create_test_dictionary() -> (TempDir, Dictionary) {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");

        let file_manager = Arc::new(FileManager::open(&db_path, true).unwrap());
        let buffer_pool = Arc::new(BufferPool::new(100, file_manager));

        let dict = Dictionary::new(buffer_pool);
        (temp_dir, dict)
    }

    #[test]
    fn test_dictionary_encode_decode() -> Result<()> {
        let (_temp_dir, dict) = create_test_dictionary();

        let term = Term::iri("http://example.org/resource");
        let id = dict.encode(&term)?;

        // Encoding again returns same ID
        let id2 = dict.encode(&term)?;
        assert_eq!(id, id2);

        // Decoding returns original term
        let decoded = dict.decode(id)?;
        assert_eq!(decoded, Some(term));

        Ok(())
    }

    #[test]
    fn test_dictionary_lookup() -> Result<()> {
        let (_temp_dir, dict) = create_test_dictionary();

        let term1 = Term::literal("test");
        let term2 = Term::literal("not-exists");

        dict.encode(&term1)?;

        assert!(dict.lookup(&term1)?.is_some());
        assert!(dict.lookup(&term2)?.is_none());

        Ok(())
    }

    #[test]
    fn test_dictionary_contains() -> Result<()> {
        let (_temp_dir, dict) = create_test_dictionary();

        let term = Term::blank_node("b0");

        assert!(!dict.contains(&term)?);
        dict.encode(&term)?;
        assert!(dict.contains(&term)?);

        Ok(())
    }

    #[test]
    fn test_dictionary_size() -> Result<()> {
        let (_temp_dir, dict) = create_test_dictionary();

        assert_eq!(dict.size(), 0);

        dict.encode(&Term::iri("http://a.com"))?;
        assert_eq!(dict.size(), 1);

        dict.encode(&Term::iri("http://b.com"))?;
        assert_eq!(dict.size(), 2);

        // Encoding same term doesn't increase size
        dict.encode(&Term::iri("http://a.com"))?;
        assert_eq!(dict.size(), 2);

        Ok(())
    }

    #[test]
    fn test_dictionary_multiple_types() -> Result<()> {
        let (_temp_dir, dict) = create_test_dictionary();

        let iri = Term::iri("http://example.org");
        let literal = Term::literal("value");
        let literal_lang = Term::literal_with_lang("valeur", "fr");
        let literal_dt = Term::literal_with_datatype("42", "http://www.w3.org/2001/XMLSchema#integer");
        let blank = Term::blank_node("b1");

        let id1 = dict.encode(&iri)?;
        let id2 = dict.encode(&literal)?;
        let id3 = dict.encode(&literal_lang)?;
        let id4 = dict.encode(&literal_dt)?;
        let id5 = dict.encode(&blank)?;

        // All different
        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
        assert_ne!(id3, id4);
        assert_ne!(id4, id5);

        // All decodable
        assert_eq!(dict.decode(id1)?, Some(iri));
        assert_eq!(dict.decode(id2)?, Some(literal));
        assert_eq!(dict.decode(id3)?, Some(literal_lang));
        assert_eq!(dict.decode(id4)?, Some(literal_dt));
        assert_eq!(dict.decode(id5)?, Some(blank));

        Ok(())
    }
}
