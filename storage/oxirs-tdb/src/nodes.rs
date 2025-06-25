//! # Node Table Implementation for TDB Storage
//!
//! Manages the encoding, compression, and storage of RDF terms (IRIs, literals, blank nodes).
//! Provides efficient serialization with dictionary compression and term interning.

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt::{self, Debug, Display};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, RwLock};

/// Node identifier type - unique ID for each term
pub type NodeId = u64;

/// Invalid node ID constant
pub const INVALID_NODE_ID: NodeId = 0;

/// RDF term representation
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Term {
    /// IRI/URI reference
    Iri(String),
    /// Literal value with optional datatype and language
    Literal {
        value: String,
        datatype: Option<String>,
        language: Option<String>,
    },
    /// Blank node with identifier
    BlankNode(String),
    /// Variable (for query patterns)
    Variable(String),
}

impl Term {
    /// Create a new IRI term
    pub fn iri<S: Into<String>>(iri: S) -> Self {
        Term::Iri(iri.into())
    }

    /// Create a new literal term
    pub fn literal<S: Into<String>>(value: S) -> Self {
        Term::Literal {
            value: value.into(),
            datatype: None,
            language: None,
        }
    }

    /// Create a new typed literal
    pub fn typed_literal<S: Into<String>, T: Into<String>>(value: S, datatype: T) -> Self {
        Term::Literal {
            value: value.into(),
            datatype: Some(datatype.into()),
            language: None,
        }
    }

    /// Create a new language-tagged literal
    pub fn lang_literal<S: Into<String>, L: Into<String>>(value: S, language: L) -> Self {
        Term::Literal {
            value: value.into(),
            datatype: None,
            language: Some(language.into()),
        }
    }

    /// Create a new blank node
    pub fn blank_node<S: Into<String>>(id: S) -> Self {
        Term::BlankNode(id.into())
    }

    /// Create a new variable
    pub fn variable<S: Into<String>>(name: S) -> Self {
        Term::Variable(name.into())
    }

    /// Get the term type as a string
    pub fn term_type(&self) -> &'static str {
        match self {
            Term::Iri(_) => "iri",
            Term::Literal { .. } => "literal",
            Term::BlankNode(_) => "blank_node",
            Term::Variable(_) => "variable",
        }
    }

    /// Check if the term is an IRI
    pub fn is_iri(&self) -> bool {
        matches!(self, Term::Iri(_))
    }

    /// Check if the term is a literal
    pub fn is_literal(&self) -> bool {
        matches!(self, Term::Literal { .. })
    }

    /// Check if the term is a blank node
    pub fn is_blank_node(&self) -> bool {
        matches!(self, Term::BlankNode(_))
    }

    /// Check if the term is a variable
    pub fn is_variable(&self) -> bool {
        matches!(self, Term::Variable(_))
    }

    /// Get the string representation of the term
    pub fn as_str(&self) -> &str {
        match self {
            Term::Iri(iri) => iri,
            Term::Literal { value, .. } => value,
            Term::BlankNode(id) => id,
            Term::Variable(name) => name,
        }
    }

    /// Estimate the serialized size in bytes
    pub fn estimated_size(&self) -> usize {
        match self {
            Term::Iri(iri) => 1 + iri.len(),
            Term::Literal {
                value,
                datatype,
                language,
            } => {
                1 + value.len()
                    + datatype.as_ref().map_or(0, |dt| dt.len())
                    + language.as_ref().map_or(0, |lang| lang.len())
            }
            Term::BlankNode(id) => 1 + id.len(),
            Term::Variable(name) => 1 + name.len(),
        }
    }
}

impl Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Term::Iri(iri) => write!(f, "<{}>", iri),
            Term::Literal {
                value,
                datatype,
                language,
            } => {
                write!(f, "\"{}\"", value)?;
                if let Some(lang) = language {
                    write!(f, "@{}", lang)?;
                } else if let Some(dt) = datatype {
                    write!(f, "^^<{}>", dt)?;
                }
                Ok(())
            }
            Term::BlankNode(id) => write!(f, "_:{}", id),
            Term::Variable(name) => write!(f, "?{}", name),
        }
    }
}

/// Node encoding for storage optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodedNode {
    /// Node type discriminator
    pub node_type: u8,
    /// Encoded data
    pub data: Vec<u8>,
    /// Compression type used
    pub compression: CompressionType,
    /// Original size before compression
    pub original_size: u32,
}

/// Compression types supported
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionType {
    None = 0,
    Dictionary = 1,
    Prefix = 2,
    Delta = 3,
    Huffman = 4,
    LZ4 = 5,
}

impl EncodedNode {
    /// Create a new encoded node
    pub fn new(
        node_type: u8,
        data: Vec<u8>,
        compression: CompressionType,
        original_size: u32,
    ) -> Self {
        Self {
            node_type,
            data,
            compression,
            original_size,
        }
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f32 {
        if self.original_size == 0 {
            0.0
        } else {
            self.data.len() as f32 / self.original_size as f32
        }
    }
}

/// Dictionary for string compression
#[derive(Debug, Clone)]
pub struct StringDictionary {
    /// String to ID mapping
    string_to_id: HashMap<String, u32>,
    /// ID to string mapping
    id_to_string: HashMap<u32, String>,
    /// Next available ID
    next_id: u32,
    /// Reference counts for garbage collection
    ref_counts: HashMap<u32, u32>,
}

impl StringDictionary {
    /// Create a new string dictionary
    pub fn new() -> Self {
        Self {
            string_to_id: HashMap::new(),
            id_to_string: HashMap::new(),
            next_id: 1, // 0 is reserved for null/invalid
            ref_counts: HashMap::new(),
        }
    }

    /// Intern a string and return its ID
    pub fn intern(&mut self, s: &str) -> u32 {
        if let Some(&id) = self.string_to_id.get(s) {
            // Increment reference count
            *self.ref_counts.entry(id).or_insert(0) += 1;
            id
        } else {
            let id = self.next_id;
            self.next_id += 1;

            self.string_to_id.insert(s.to_string(), id);
            self.id_to_string.insert(id, s.to_string());
            self.ref_counts.insert(id, 1);

            id
        }
    }

    /// Get string by ID
    pub fn get_string(&self, id: u32) -> Option<&String> {
        self.id_to_string.get(&id)
    }

    /// Get ID by string
    pub fn get_id(&self, s: &str) -> Option<u32> {
        self.string_to_id.get(s).copied()
    }

    /// Decrement reference count and potentially remove string
    pub fn release(&mut self, id: u32) -> bool {
        if let Some(count) = self.ref_counts.get_mut(&id) {
            *count -= 1;
            if *count == 0 {
                // Remove from dictionary
                if let Some(string) = self.id_to_string.remove(&id) {
                    self.string_to_id.remove(&string);
                }
                self.ref_counts.remove(&id);
                return true;
            }
        }
        false
    }

    /// Get dictionary statistics
    pub fn stats(&self) -> (usize, usize, u32) {
        (
            self.string_to_id.len(),
            self.id_to_string.len(),
            self.next_id,
        )
    }
}

impl Default for StringDictionary {
    fn default() -> Self {
        Self::new()
    }
}

/// Node table configuration
#[derive(Debug, Clone)]
pub struct NodeTableConfig {
    /// Enable compression
    pub enable_compression: bool,
    /// Compression threshold in bytes
    pub compression_threshold: usize,
    /// Default compression type
    pub default_compression: CompressionType,
    /// Enable term interning
    pub enable_interning: bool,
    /// Cache size for frequently accessed nodes
    pub cache_size: usize,
    /// Enable prefix compression for IRIs
    pub enable_prefix_compression: bool,
}

impl Default for NodeTableConfig {
    fn default() -> Self {
        Self {
            enable_compression: true,
            compression_threshold: 32,
            default_compression: CompressionType::Dictionary,
            enable_interning: true,
            cache_size: 10000,
            enable_prefix_compression: true,
        }
    }
}

/// Node table statistics
#[derive(Debug, Clone, Default, Serialize)]
pub struct NodeTableStats {
    /// Total nodes stored
    pub total_nodes: u64,
    /// IRI nodes count
    pub iri_nodes: u64,
    /// Literal nodes count
    pub literal_nodes: u64,
    /// Blank node count
    pub blank_nodes: u64,
    /// Variable count
    pub variables: u64,
    /// Compressed nodes count
    pub compressed_nodes: u64,
    /// Total bytes stored
    pub total_bytes: u64,
    /// Compressed bytes
    pub compressed_bytes: u64,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Dictionary size
    pub dictionary_size: usize,
}

impl NodeTableStats {
    /// Calculate compression ratio
    pub fn compression_ratio(&self) -> f32 {
        if self.total_bytes == 0 {
            0.0
        } else {
            self.compressed_bytes as f32 / self.total_bytes as f32
        }
    }

    /// Calculate cache hit ratio
    pub fn cache_hit_ratio(&self) -> f32 {
        let total_requests = self.cache_hits + self.cache_misses;
        if total_requests == 0 {
            0.0
        } else {
            self.cache_hits as f32 / total_requests as f32
        }
    }
}

/// Node table implementation
pub struct NodeTable {
    /// Configuration
    config: NodeTableConfig,
    /// Node ID to encoded node mapping
    nodes: Arc<RwLock<HashMap<NodeId, EncodedNode>>>,
    /// Term to node ID mapping
    term_to_id: Arc<RwLock<HashMap<Term, NodeId>>>,
    /// Next available node ID
    next_node_id: Arc<RwLock<NodeId>>,
    /// String dictionary for compression
    dictionary: Arc<RwLock<StringDictionary>>,
    /// Frequently accessed nodes cache
    cache: Arc<RwLock<HashMap<NodeId, Term>>>,
    /// Common IRI prefixes for compression
    iri_prefixes: Arc<RwLock<HashMap<String, u32>>>,
    /// Statistics
    stats: Arc<Mutex<NodeTableStats>>,
}

impl NodeTable {
    /// Create a new node table
    pub fn new() -> Self {
        Self::with_config(NodeTableConfig::default())
    }

    /// Create a new node table with configuration
    pub fn with_config(config: NodeTableConfig) -> Self {
        let mut iri_prefixes = HashMap::new();

        // Add common RDF prefixes
        iri_prefixes.insert("http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string(), 1);
        iri_prefixes.insert("http://www.w3.org/2000/01/rdf-schema#".to_string(), 2);
        iri_prefixes.insert("http://www.w3.org/2001/XMLSchema#".to_string(), 3);
        iri_prefixes.insert("http://www.w3.org/2002/07/owl#".to_string(), 4);

        Self {
            config,
            nodes: Arc::new(RwLock::new(HashMap::new())),
            term_to_id: Arc::new(RwLock::new(HashMap::new())),
            next_node_id: Arc::new(RwLock::new(1)), // 0 is reserved for invalid
            dictionary: Arc::new(RwLock::new(StringDictionary::new())),
            cache: Arc::new(RwLock::new(HashMap::new())),
            iri_prefixes: Arc::new(RwLock::new(iri_prefixes)),
            stats: Arc::new(Mutex::new(NodeTableStats::default())),
        }
    }

    /// Store a term and return its node ID
    pub fn store_term(&self, term: &Term) -> Result<NodeId> {
        // Check if term already exists
        if let Some(node_id) = self.get_existing_node_id(term)? {
            self.update_cache(node_id, term.clone())?;
            return Ok(node_id);
        }

        // Allocate new node ID
        let node_id = self.allocate_node_id()?;

        // Encode the term
        let encoded = self.encode_term(term)?;

        // Store encoded node
        {
            let mut nodes = self
                .nodes
                .write()
                .map_err(|_| anyhow!("Failed to acquire nodes lock"))?;
            nodes.insert(node_id, encoded);
        }

        // Update term to ID mapping
        {
            let mut term_to_id = self
                .term_to_id
                .write()
                .map_err(|_| anyhow!("Failed to acquire term_to_id lock"))?;
            term_to_id.insert(term.clone(), node_id);
        }

        // Update cache
        self.update_cache(node_id, term.clone())?;

        // Update statistics
        self.update_stats_on_store(term)?;

        Ok(node_id)
    }

    /// Retrieve a term by node ID
    pub fn get_term(&self, node_id: NodeId) -> Result<Option<Term>> {
        // Check cache first
        if let Some(term) = self.get_from_cache(node_id)? {
            self.update_stats_cache_hit();
            return Ok(Some(term));
        }

        self.update_stats_cache_miss();

        // Get encoded node
        let encoded = {
            let nodes = self
                .nodes
                .read()
                .map_err(|_| anyhow!("Failed to acquire nodes lock"))?;
            nodes.get(&node_id).cloned()
        };

        if let Some(encoded) = encoded {
            let term = self.decode_term(&encoded)?;
            self.update_cache(node_id, term.clone())?;
            Ok(Some(term))
        } else {
            Ok(None)
        }
    }

    /// Get node ID for a term without storing
    pub fn get_node_id(&self, term: &Term) -> Result<Option<NodeId>> {
        self.get_existing_node_id(term)
    }

    /// Remove a term from the table
    pub fn remove_term(&self, term: &Term) -> Result<bool> {
        if let Some(node_id) = self.get_existing_node_id(term)? {
            self.remove_node(node_id)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Remove a node by ID
    pub fn remove_node(&self, node_id: NodeId) -> Result<()> {
        // Remove from nodes
        let encoded = {
            let mut nodes = self
                .nodes
                .write()
                .map_err(|_| anyhow!("Failed to acquire nodes lock"))?;
            nodes.remove(&node_id)
        };

        if let Some(encoded) = encoded {
            // Decode to get the term for removal from term_to_id mapping
            let term = self.decode_term(&encoded)?;

            // Remove from term_to_id mapping
            {
                let mut term_to_id = self
                    .term_to_id
                    .write()
                    .map_err(|_| anyhow!("Failed to acquire term_to_id lock"))?;
                term_to_id.remove(&term);
            }

            // Remove from cache
            {
                let mut cache = self
                    .cache
                    .write()
                    .map_err(|_| anyhow!("Failed to acquire cache lock"))?;
                cache.remove(&node_id);
            }

            // Update statistics
            self.update_stats_on_remove(&term)?;
        }

        Ok(())
    }

    /// Get node table statistics
    pub fn get_stats(&self) -> Result<NodeTableStats> {
        let stats = self
            .stats
            .lock()
            .map_err(|_| anyhow!("Failed to acquire stats lock"))?;
        Ok(stats.clone())
    }

    /// Compact the node table (cleanup unused dictionary entries)
    pub fn compact(&self) -> Result<usize> {
        let mut dictionary = self
            .dictionary
            .write()
            .map_err(|_| anyhow!("Failed to acquire dictionary lock"))?;

        let (before_size, _, _) = dictionary.stats();

        // In a real implementation, we would scan all nodes and rebuild the dictionary
        // For now, this is a placeholder

        let (after_size, _, _) = dictionary.stats();
        let cleaned = before_size.saturating_sub(after_size);

        // Update stats
        if let Ok(mut stats) = self.stats.lock() {
            stats.dictionary_size = after_size;
        }

        Ok(cleaned)
    }

    /// Clear all nodes
    pub fn clear(&self) -> Result<()> {
        {
            let mut nodes = self
                .nodes
                .write()
                .map_err(|_| anyhow!("Failed to acquire nodes lock"))?;
            nodes.clear();
        }

        {
            let mut term_to_id = self
                .term_to_id
                .write()
                .map_err(|_| anyhow!("Failed to acquire term_to_id lock"))?;
            term_to_id.clear();
        }

        {
            let mut cache = self
                .cache
                .write()
                .map_err(|_| anyhow!("Failed to acquire cache lock"))?;
            cache.clear();
        }

        {
            let mut dictionary = self
                .dictionary
                .write()
                .map_err(|_| anyhow!("Failed to acquire dictionary lock"))?;
            *dictionary = StringDictionary::new();
        }

        {
            let mut next_node_id = self
                .next_node_id
                .write()
                .map_err(|_| anyhow!("Failed to acquire next_node_id lock"))?;
            *next_node_id = 1;
        }

        {
            let mut stats = self
                .stats
                .lock()
                .map_err(|_| anyhow!("Failed to acquire stats lock"))?;
            *stats = NodeTableStats::default();
        }

        Ok(())
    }

    /// Get the total number of stored nodes
    pub fn len(&self) -> Result<usize> {
        let nodes = self
            .nodes
            .read()
            .map_err(|_| anyhow!("Failed to acquire nodes lock"))?;
        Ok(nodes.len())
    }

    /// Check if the table is empty
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    // Private helper methods

    fn get_existing_node_id(&self, term: &Term) -> Result<Option<NodeId>> {
        let term_to_id = self
            .term_to_id
            .read()
            .map_err(|_| anyhow!("Failed to acquire term_to_id lock"))?;
        Ok(term_to_id.get(term).copied())
    }

    fn allocate_node_id(&self) -> Result<NodeId> {
        let mut next_node_id = self
            .next_node_id
            .write()
            .map_err(|_| anyhow!("Failed to acquire next_node_id lock"))?;
        let id = *next_node_id;
        *next_node_id += 1;
        Ok(id)
    }

    fn encode_term(&self, term: &Term) -> Result<EncodedNode> {
        let node_type = match term {
            Term::Iri(_) => 1,
            Term::Literal { .. } => 2,
            Term::BlankNode(_) => 3,
            Term::Variable(_) => 4,
        };

        let serialized =
            bincode::serialize(term).map_err(|e| anyhow!("Failed to serialize term: {}", e))?;

        let original_size = serialized.len() as u32;

        // Apply compression if enabled and beneficial
        let (data, compression) = if self.config.enable_compression
            && serialized.len() > self.config.compression_threshold
        {
            self.compress_data(&serialized, term)?
        } else {
            (serialized, CompressionType::None)
        };

        Ok(EncodedNode::new(
            node_type,
            data,
            compression,
            original_size,
        ))
    }

    fn decode_term(&self, encoded: &EncodedNode) -> Result<Term> {
        let data = match encoded.compression {
            CompressionType::None => encoded.data.clone(),
            _ => self.decompress_data(&encoded.data, encoded.compression)?,
        };

        bincode::deserialize(&data).map_err(|e| anyhow!("Failed to deserialize term: {}", e))
    }

    fn compress_data(&self, data: &[u8], term: &Term) -> Result<(Vec<u8>, CompressionType)> {
        match self.config.default_compression {
            CompressionType::Dictionary => {
                // Dictionary compression for strings
                if let Ok(dict_data) = self.dictionary_compress(data, term) {
                    Ok((dict_data, CompressionType::Dictionary))
                } else {
                    Ok((data.to_vec(), CompressionType::None))
                }
            }
            CompressionType::Prefix => {
                // Prefix compression for IRIs
                if let Term::Iri(iri) = term {
                    if let Ok(prefix_data) = self.prefix_compress(iri) {
                        Ok((prefix_data, CompressionType::Prefix))
                    } else {
                        Ok((data.to_vec(), CompressionType::None))
                    }
                } else {
                    Ok((data.to_vec(), CompressionType::None))
                }
            }
            _ => Ok((data.to_vec(), CompressionType::None)), // Other compression types not implemented
        }
    }

    fn decompress_data(&self, data: &[u8], compression: CompressionType) -> Result<Vec<u8>> {
        match compression {
            CompressionType::None => Ok(data.to_vec()),
            CompressionType::Dictionary => self.dictionary_decompress(data),
            CompressionType::Prefix => self.prefix_decompress(data),
            _ => Err(anyhow!("Unsupported compression type: {:?}", compression)),
        }
    }

    fn dictionary_compress(&self, data: &[u8], term: &Term) -> Result<Vec<u8>> {
        // Simplified dictionary compression
        let term_str = match term {
            Term::Iri(iri) => iri,
            Term::Literal { value, .. } => value,
            Term::BlankNode(id) => id,
            Term::Variable(name) => name,
        };

        let mut dictionary = self
            .dictionary
            .write()
            .map_err(|_| anyhow!("Failed to acquire dictionary lock"))?;

        let dict_id = dictionary.intern(term_str);
        Ok(dict_id.to_le_bytes().to_vec())
    }

    fn dictionary_decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        if data.len() != 4 {
            return Err(anyhow!("Invalid dictionary compressed data length"));
        }

        let dict_id = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let dictionary = self
            .dictionary
            .read()
            .map_err(|_| anyhow!("Failed to acquire dictionary lock"))?;

        if let Some(string) = dictionary.get_string(dict_id) {
            bincode::serialize(string)
                .map_err(|e| anyhow!("Failed to serialize dictionary string: {}", e))
        } else {
            Err(anyhow!("Dictionary ID {} not found", dict_id))
        }
    }

    fn prefix_compress(&self, iri: &str) -> Result<Vec<u8>> {
        let iri_prefixes = self
            .iri_prefixes
            .read()
            .map_err(|_| anyhow!("Failed to acquire IRI prefixes lock"))?;

        for (prefix, &prefix_id) in iri_prefixes.iter() {
            if iri.starts_with(prefix) {
                let suffix = &iri[prefix.len()..];
                let mut result = Vec::new();
                result.extend_from_slice(&prefix_id.to_le_bytes());
                result.extend_from_slice(suffix.as_bytes());
                return Ok(result);
            }
        }

        Err(anyhow!("No matching prefix found for IRI: {}", iri))
    }

    fn prefix_decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        if data.len() < 4 {
            return Err(anyhow!("Invalid prefix compressed data length"));
        }

        let prefix_id = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let suffix = std::str::from_utf8(&data[4..])
            .map_err(|e| anyhow!("Invalid UTF-8 in suffix: {}", e))?;

        let iri_prefixes = self
            .iri_prefixes
            .read()
            .map_err(|_| anyhow!("Failed to acquire IRI prefixes lock"))?;

        let prefix = iri_prefixes
            .iter()
            .find(|(_, &id)| id == prefix_id)
            .map(|(prefix, _)| prefix)
            .ok_or_else(|| anyhow!("Prefix ID {} not found", prefix_id))?;

        let full_iri = format!("{}{}", prefix, suffix);
        bincode::serialize(&full_iri)
            .map_err(|e| anyhow!("Failed to serialize reconstructed IRI: {}", e))
    }

    fn get_from_cache(&self, node_id: NodeId) -> Result<Option<Term>> {
        let cache = self
            .cache
            .read()
            .map_err(|_| anyhow!("Failed to acquire cache lock"))?;
        Ok(cache.get(&node_id).cloned())
    }

    fn update_cache(&self, node_id: NodeId, term: Term) -> Result<()> {
        let mut cache = self
            .cache
            .write()
            .map_err(|_| anyhow!("Failed to acquire cache lock"))?;

        // Simple LRU eviction if cache is full
        if cache.len() >= self.config.cache_size {
            // Remove oldest entry (this is a simplified implementation)
            if let Some((&oldest_id, _)) = cache.iter().next() {
                let oldest_id = oldest_id;
                cache.remove(&oldest_id);
            }
        }

        cache.insert(node_id, term);
        Ok(())
    }

    fn update_stats_on_store(&self, term: &Term) -> Result<()> {
        let mut stats = self
            .stats
            .lock()
            .map_err(|_| anyhow!("Failed to acquire stats lock"))?;

        stats.total_nodes += 1;
        match term {
            Term::Iri(_) => stats.iri_nodes += 1,
            Term::Literal { .. } => stats.literal_nodes += 1,
            Term::BlankNode(_) => stats.blank_nodes += 1,
            Term::Variable(_) => stats.variables += 1,
        }

        let estimated_size = term.estimated_size() as u64;
        stats.total_bytes += estimated_size;

        Ok(())
    }

    fn update_stats_on_remove(&self, term: &Term) -> Result<()> {
        let mut stats = self
            .stats
            .lock()
            .map_err(|_| anyhow!("Failed to acquire stats lock"))?;

        stats.total_nodes = stats.total_nodes.saturating_sub(1);
        match term {
            Term::Iri(_) => stats.iri_nodes = stats.iri_nodes.saturating_sub(1),
            Term::Literal { .. } => stats.literal_nodes = stats.literal_nodes.saturating_sub(1),
            Term::BlankNode(_) => stats.blank_nodes = stats.blank_nodes.saturating_sub(1),
            Term::Variable(_) => stats.variables = stats.variables.saturating_sub(1),
        }

        let estimated_size = term.estimated_size() as u64;
        stats.total_bytes = stats.total_bytes.saturating_sub(estimated_size);

        Ok(())
    }

    fn update_stats_cache_hit(&self) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.cache_hits += 1;
        }
    }

    fn update_stats_cache_miss(&self) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.cache_misses += 1;
        }
    }
}

impl Default for NodeTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_term_creation() {
        let iri = Term::iri("http://example.org/person");
        assert!(iri.is_iri());
        assert_eq!(iri.as_str(), "http://example.org/person");

        let literal = Term::literal("Hello World");
        assert!(literal.is_literal());
        assert_eq!(literal.as_str(), "Hello World");

        let typed_literal = Term::typed_literal("42", "http://www.w3.org/2001/XMLSchema#integer");
        assert!(typed_literal.is_literal());

        let lang_literal = Term::lang_literal("Bonjour", "fr");
        assert!(lang_literal.is_literal());

        let blank_node = Term::blank_node("b1");
        assert!(blank_node.is_blank_node());
        assert_eq!(blank_node.as_str(), "b1");

        let variable = Term::variable("x");
        assert!(variable.is_variable());
        assert_eq!(variable.as_str(), "x");
    }

    #[test]
    fn test_node_table_basic_operations() {
        let node_table = NodeTable::new();

        let term1 = Term::iri("http://example.org/person");
        let term2 = Term::literal("John Doe");

        // Store terms
        let node_id1 = node_table.store_term(&term1).unwrap();
        let node_id2 = node_table.store_term(&term2).unwrap();

        assert_ne!(node_id1, node_id2);

        // Retrieve terms
        let retrieved1 = node_table.get_term(node_id1).unwrap().unwrap();
        let retrieved2 = node_table.get_term(node_id2).unwrap().unwrap();

        assert_eq!(retrieved1, term1);
        assert_eq!(retrieved2, term2);

        // Store same term again (should return same ID)
        let node_id1_again = node_table.store_term(&term1).unwrap();
        assert_eq!(node_id1, node_id1_again);

        // Get node ID without storing
        let found_id = node_table.get_node_id(&term1).unwrap().unwrap();
        assert_eq!(found_id, node_id1);
    }

    #[test]
    fn test_node_table_removal() {
        let node_table = NodeTable::new();

        let term = Term::iri("http://example.org/test");
        let node_id = node_table.store_term(&term).unwrap();

        // Verify term exists
        assert!(node_table.get_term(node_id).unwrap().is_some());

        // Remove term
        assert!(node_table.remove_term(&term).unwrap());

        // Verify term is gone
        assert!(node_table.get_term(node_id).unwrap().is_none());

        // Try to remove again
        assert!(!node_table.remove_term(&term).unwrap());
    }

    #[test]
    fn test_node_table_stats() {
        let node_table = NodeTable::new();

        let terms = vec![
            Term::iri("http://example.org/person"),
            Term::literal("John Doe"),
            Term::blank_node("b1"),
            Term::variable("x"),
        ];

        for term in &terms {
            node_table.store_term(term).unwrap();
        }

        let stats = node_table.get_stats().unwrap();
        assert_eq!(stats.total_nodes, 4);
        assert_eq!(stats.iri_nodes, 1);
        assert_eq!(stats.literal_nodes, 1);
        assert_eq!(stats.blank_nodes, 1);
        assert_eq!(stats.variables, 1);
    }

    #[test]
    fn test_string_dictionary() {
        let mut dict = StringDictionary::new();

        let id1 = dict.intern("hello");
        let id2 = dict.intern("world");
        let id3 = dict.intern("hello"); // Same string

        assert_eq!(id1, id3); // Same ID for same string
        assert_ne!(id1, id2); // Different IDs for different strings

        assert_eq!(dict.get_string(id1), Some(&"hello".to_string()));
        assert_eq!(dict.get_string(id2), Some(&"world".to_string()));

        assert_eq!(dict.get_id("hello"), Some(id1));
        assert_eq!(dict.get_id("world"), Some(id2));
    }
}
