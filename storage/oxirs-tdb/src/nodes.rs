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

use crate::compression::{
    AdaptiveCompressor, AdvancedCompressionType, ColumnStoreCompressor, CompressedData,
    CompressionMetadata, RunLengthEncoder,
};

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
    // Advanced compression types
    RunLength = 10,
    BitmapWAH = 11,
    BitmapRoaring = 12,
    FrameOfReference = 14,
    AdaptiveDictionary = 15,
    ColumnStore = 16,
    Adaptive = 17,
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

/// Dictionary for string compression with enhanced features per TODO.md Phase 1.3.2
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
    /// Configuration for advanced features
    config: StringDictionaryConfig,
}

/// Configuration for string dictionary advanced features
#[derive(Debug, Clone)]
pub struct StringDictionaryConfig {
    /// Enable automatic garbage collection
    pub enable_gc: bool,
    /// Garbage collection threshold (trigger when ref_counts exceeds this)
    pub gc_threshold: usize,
    /// Enable persistence to disk
    pub enable_persistence: bool,
    /// Persistence file path
    pub persistence_path: Option<String>,
}

impl Default for StringDictionaryConfig {
    fn default() -> Self {
        Self {
            enable_gc: true,
            gc_threshold: 10000,
            enable_persistence: false,
            persistence_path: None,
        }
    }
}

impl StringDictionary {
    /// Create a new string dictionary with default configuration
    pub fn new() -> Self {
        Self::with_config(StringDictionaryConfig::default())
    }

    /// Create a new string dictionary with custom configuration
    pub fn with_config(config: StringDictionaryConfig) -> Self {
        Self {
            string_to_id: HashMap::new(),
            id_to_string: HashMap::new(),
            next_id: 1, // 0 is reserved for null/invalid
            ref_counts: HashMap::new(),
            config,
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

            // Check if garbage collection is needed
            if self.config.enable_gc && self.ref_counts.len() >= self.config.gc_threshold {
                self.garbage_collect();
            }

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

    /// Get detailed statistics about the dictionary
    pub fn detailed_stats(&self) -> StringDictionaryStats {
        let string_count = self.string_to_id.len();
        let ref_count_entries = self.ref_counts.len();

        // Calculate total memory usage (approximate)
        let string_memory: usize = self.id_to_string.values().map(|s| s.len()).sum();

        let map_overhead =
            string_count * (std::mem::size_of::<String>() + std::mem::size_of::<u32>()) * 2;
        let ref_count_overhead = ref_count_entries * std::mem::size_of::<u32>() * 2;

        StringDictionaryStats {
            string_count,
            total_memory_bytes: string_memory + map_overhead + ref_count_overhead,
            ref_count_entries,
            gc_enabled: self.config.enable_gc,
        }
    }

    /// Perform garbage collection
    ///
    /// Removes strings with zero reference counts from the dictionary.
    /// Returns the number of strings removed.
    pub fn garbage_collect(&mut self) -> usize {
        if !self.config.enable_gc {
            return 0;
        }

        let mut removed_count = 0;
        let ids_to_remove: Vec<u32> = self
            .ref_counts
            .iter()
            .filter_map(|(&id, &count)| if count == 0 { Some(id) } else { None })
            .collect();

        for id in ids_to_remove {
            if let Some(string) = self.id_to_string.remove(&id) {
                self.string_to_id.remove(&string);
                self.ref_counts.remove(&id);
                removed_count += 1;
            }
        }

        removed_count
    }

    /// Get current reference count for an ID
    pub fn ref_count(&self, id: u32) -> u32 {
        self.ref_counts.get(&id).copied().unwrap_or(0)
    }

    /// Clear all entries from the dictionary
    pub fn clear(&mut self) {
        self.string_to_id.clear();
        self.id_to_string.clear();
        self.ref_counts.clear();
        self.next_id = 1;
    }

    /// Save dictionary to disk (if persistence is enabled)
    pub fn persist(&self) -> Result<()> {
        if !self.config.enable_persistence {
            return Ok(());
        }

        let path = self
            .config
            .persistence_path
            .as_ref()
            .ok_or_else(|| anyhow!("Persistence enabled but no path specified"))?;

        let snapshot = StringDictionarySnapshot {
            string_to_id: self.string_to_id.clone(),
            id_to_string: self.id_to_string.clone(),
            ref_counts: self.ref_counts.clone(),
            next_id: self.next_id,
        };

        let serialized = bincode::serialize(&snapshot)
            .map_err(|e| anyhow!("Failed to serialize dictionary: {}", e))?;

        std::fs::write(path, serialized)
            .map_err(|e| anyhow!("Failed to write dictionary to {}: {}", path, e))
    }

    /// Load dictionary from disk (if persistence is enabled)
    pub fn load(&mut self) -> Result<()> {
        if !self.config.enable_persistence {
            return Ok(());
        }

        let path = self
            .config
            .persistence_path
            .as_ref()
            .ok_or_else(|| anyhow!("Persistence enabled but no path specified"))?;

        if !std::path::Path::new(path).exists() {
            return Ok(()); // No existing dictionary to load
        }

        let data = std::fs::read(path)
            .map_err(|e| anyhow!("Failed to read dictionary from {}: {}", path, e))?;

        let snapshot: StringDictionarySnapshot = bincode::deserialize(&data)
            .map_err(|e| anyhow!("Failed to deserialize dictionary: {}", e))?;

        // Restore state
        self.string_to_id = snapshot.string_to_id;
        self.id_to_string = snapshot.id_to_string;
        self.ref_counts = snapshot.ref_counts;
        self.next_id = snapshot.next_id;

        Ok(())
    }
}

impl Default for StringDictionary {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the string dictionary
#[derive(Debug, Clone)]
pub struct StringDictionaryStats {
    /// Number of unique strings stored
    pub string_count: usize,
    /// Total memory usage in bytes (approximate)
    pub total_memory_bytes: usize,
    /// Number of reference count entries
    pub ref_count_entries: usize,
    /// Whether garbage collection is enabled
    pub gc_enabled: bool,
}

/// Serializable snapshot of dictionary state for persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StringDictionarySnapshot {
    string_to_id: HashMap<String, u32>,
    id_to_string: HashMap<u32, String>,
    ref_counts: HashMap<u32, u32>,
    next_id: u32,
}

/// Global string dictionary instance for project-wide string interning
static GLOBAL_STRING_DICT: once_cell::sync::Lazy<std::sync::Mutex<StringDictionary>> =
    once_cell::sync::Lazy::new(|| std::sync::Mutex::new(StringDictionary::new()));

/// Get mutable reference to the global string dictionary
pub fn global_string_dict() -> std::sync::MutexGuard<'static, StringDictionary> {
    GLOBAL_STRING_DICT.lock().unwrap()
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
    /// Enable advanced compression algorithms
    pub enable_advanced_compression: bool,
    /// Enable column-store optimizations
    pub enable_column_store: bool,
    /// Adaptive compression threshold
    pub adaptive_threshold: f64,
}

impl Default for NodeTableConfig {
    fn default() -> Self {
        Self {
            enable_compression: true,
            compression_threshold: 32,
            default_compression: CompressionType::Adaptive,
            enable_interning: true,
            cache_size: 10000,
            enable_prefix_compression: true,
            enable_advanced_compression: true,
            enable_column_store: true,
            adaptive_threshold: 0.8,
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
    /// Advanced adaptive compressor
    adaptive_compressor: Arc<AdaptiveCompressor>,
    /// Column-store compressor
    column_compressor: Arc<RwLock<ColumnStoreCompressor>>,
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

        // Initialize advanced compressors
        let adaptive_compressor = Arc::new(AdaptiveCompressor::new(
            config.compression_threshold,
            config.adaptive_threshold,
        ));
        let column_compressor = Arc::new(RwLock::new(ColumnStoreCompressor::new()));

        Self {
            config,
            nodes: Arc::new(RwLock::new(HashMap::new())),
            term_to_id: Arc::new(RwLock::new(HashMap::new())),
            next_node_id: Arc::new(RwLock::new(1)), // 0 is reserved for invalid
            dictionary: Arc::new(RwLock::new(StringDictionary::new())),
            cache: Arc::new(RwLock::new(HashMap::new())),
            iri_prefixes: Arc::new(RwLock::new(iri_prefixes)),
            adaptive_compressor,
            column_compressor,
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
        // Use advanced compression if enabled
        if self.config.enable_advanced_compression {
            return self.advanced_compress_data(data, term);
        }

        // Fall back to legacy compression
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
            CompressionType::RunLength => RunLengthEncoder::decode(data),
            CompressionType::Adaptive => {
                // For adaptive compression, we need to determine the actual algorithm used
                // This is a simplified implementation - in practice, we'd store metadata
                self.adaptive_decompress_data(data)
            }
            _ => Err(anyhow!("Unsupported compression type: {:?}", compression)),
        }
    }

    fn advanced_compress_data(
        &self,
        data: &[u8],
        term: &Term,
    ) -> Result<(Vec<u8>, CompressionType)> {
        // Analyze term for column-store optimization
        if self.config.enable_column_store {
            self.analyze_term_for_column_store(term)?;
        }

        // Use adaptive compressor to select best algorithm
        match self.adaptive_compressor.compress(data) {
            Ok(compressed) => {
                let compression_type = match compressed.metadata.algorithm {
                    AdvancedCompressionType::RunLength => CompressionType::RunLength,
                    AdvancedCompressionType::Delta => CompressionType::Delta,
                    AdvancedCompressionType::FrameOfReference => CompressionType::FrameOfReference,
                    AdvancedCompressionType::AdaptiveDictionary => {
                        CompressionType::AdaptiveDictionary
                    }
                    AdvancedCompressionType::Adaptive => CompressionType::Adaptive,
                    _ => CompressionType::None,
                };

                // Only use compression if it provides significant savings
                if compressed.metadata.compression_ratio() < self.config.adaptive_threshold {
                    Ok((compressed.data, compression_type))
                } else {
                    Ok((data.to_vec(), CompressionType::None))
                }
            }
            Err(_) => {
                // Fall back to legacy compression
                self.legacy_compress_data(data, term)
            }
        }
    }

    fn adaptive_decompress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        // In a real implementation, we would store compression metadata
        // For now, try to decompress using the adaptive compressor
        let compressed_data = CompressedData {
            data: data.to_vec(),
            metadata: CompressionMetadata {
                algorithm: AdvancedCompressionType::RunLength, // Default guess
                original_size: 0,
                compressed_size: data.len() as u64,
                compression_time_us: 0,
                metadata: HashMap::new(),
            },
        };

        self.adaptive_compressor.decompress(&compressed_data)
    }

    fn legacy_compress_data(&self, data: &[u8], term: &Term) -> Result<(Vec<u8>, CompressionType)> {
        match self.config.default_compression {
            CompressionType::Dictionary => {
                if let Ok(dict_data) = self.dictionary_compress(data, term) {
                    Ok((dict_data, CompressionType::Dictionary))
                } else {
                    Ok((data.to_vec(), CompressionType::None))
                }
            }
            CompressionType::Prefix => {
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
            _ => Ok((data.to_vec(), CompressionType::None)),
        }
    }

    fn analyze_term_for_column_store(&self, term: &Term) -> Result<()> {
        if let Ok(mut compressor) = self.column_compressor.write() {
            let column_name = match term {
                Term::Iri(_) => "iri",
                Term::Literal {
                    datatype: Some(_), ..
                } => "literal_datatype",
                Term::Literal {
                    language: Some(_), ..
                } => "literal_language",
                Term::Literal { .. } => "literal_value",
                Term::BlankNode(_) => "blank_node",
                Term::Variable(_) => "variable",
            };

            // In a real implementation, we would accumulate data and analyze in batches
            let values = vec![term.as_str().to_string()];
            compressor.analyze_column(column_name, &values);
        }
        Ok(())
    }

    fn dictionary_compress(&self, data: &[u8], term: &Term) -> Result<Vec<u8>> {
        // For complex terms with metadata, don't use dictionary compression
        match term {
            Term::Literal {
                datatype: Some(_), ..
            }
            | Term::Literal {
                language: Some(_), ..
            } => {
                // Don't compress literals with datatype or language tags
                return Err(anyhow!(
                    "Complex literals not suitable for dictionary compression"
                ));
            }
            _ => {}
        }

        // Simplified dictionary compression for simple terms
        let (term_type, term_str) = match term {
            Term::Iri(iri) => (1u8, iri),
            Term::Literal { value, .. } => (2u8, value),
            Term::BlankNode(id) => (3u8, id),
            Term::Variable(name) => (4u8, name),
        };

        let mut dictionary = self
            .dictionary
            .write()
            .map_err(|_| anyhow!("Failed to acquire dictionary lock"))?;

        let dict_id = dictionary.intern(term_str);

        // Store term type + dictionary ID
        let mut result = vec![term_type];
        result.extend_from_slice(&dict_id.to_le_bytes());
        Ok(result)
    }

    fn dictionary_decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        if data.len() < 5 {
            return Err(anyhow!("Invalid dictionary compressed data length"));
        }

        // First byte indicates the term type
        let term_type = data[0];
        let dict_id = u32::from_le_bytes([data[1], data[2], data[3], data[4]]);

        let dictionary = self
            .dictionary
            .read()
            .map_err(|_| anyhow!("Failed to acquire dictionary lock"))?;

        if let Some(string) = dictionary.get_string(dict_id) {
            // Reconstruct the original Term based on type
            let term = match term_type {
                1 => Term::Iri(string.clone()),
                2 => Term::Literal {
                    value: string.clone(),
                    datatype: None,
                    language: None,
                },
                3 => Term::BlankNode(string.clone()),
                4 => Term::Variable(string.clone()),
                _ => return Err(anyhow!("Invalid term type: {}", term_type)),
            };

            bincode::serialize(&term)
                .map_err(|e| anyhow!("Failed to serialize reconstructed term: {}", e))
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
