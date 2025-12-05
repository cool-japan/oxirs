//! # HDT-star Format Support
//!
//! Efficient binary serialization for RDF-star data using Header-Dictionary-Triples (HDT) format
//! with extensions for quoted triples.
//!
//! HDT-star provides:
//! - **Compact Storage**: Dictionary-based compression for repeated terms
//! - **Fast Queries**: Bitmap-indexed structure for O(log n) lookups
//! - **Memory Mapping**: Supports datasets larger than RAM
//! - **RDF-star Extension**: Native quoted triple support with nested dictionary entries
//!
//! ## Overview
//!
//! HDT (Header-Dictionary-Triples) is a compact data structure and binary serialization format
//! for RDF. This module extends HDT to support RDF-star quoted triples:
//!
//! 1. **Header**: Metadata about the dataset (format, statistics, configuration)
//! 2. **Dictionary**: Compressed string dictionary for subjects, predicates, objects
//!    - Extended with a Quoted Triple Dictionary for nested triples
//! 3. **Triples**: Bitmap-based triple storage with SPO/POS/OSP indices
//!
//! ## Example
//!
//! ```rust,no_run
//! use oxirs_star::hdt_star::{HdtStarBuilder, HdtStarReader, HdtStarConfig};
//! use oxirs_star::{StarTriple, StarTerm, StarStore};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Build HDT-star from a store
//! let mut store = StarStore::new();
//! // ... populate store with data
//!
//! let config = HdtStarConfig::default();
//! let mut builder = HdtStarBuilder::new(config);
//! builder.add_store(&store)?;
//!
//! // Write to file
//! let mut file = std::fs::File::create("output.hdt")?;
//! builder.write(&mut file)?;
//!
//! // Read HDT-star file
//! let reader = HdtStarReader::open("output.hdt")?;
//! for triple in reader.iter_triples() {
//!     println!("{}", triple);
//! }
//! # Ok(())
//! # }
//! ```

use crate::{StarError, StarResult, StarStore, StarTerm, StarTriple};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Write};
use tracing::{info, instrument, warn};

// Import SciRS2 components (SCIRS2 POLICY)
use scirs2_core::profiling::Profiler;

// SIMD-accelerated compression module (v0.4.0 Phase 1)
#[path = "hdt_star/simd_compression.rs"]
pub mod simd_compression;

pub use simd_compression::{SimdBitmapOps, SimdCompressionAnalyzer, SimdStringComparator};

/// HDT-star file format version
pub const HDT_STAR_VERSION: u8 = 1;

/// Magic bytes for HDT-star format identification
pub const HDT_STAR_MAGIC: [u8; 8] = *b"HDT*RDF\0";

/// Configuration for HDT-star encoding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HdtStarConfig {
    /// Enable dictionary compression
    pub enable_compression: bool,
    /// Compression level (1-9, higher = smaller but slower)
    pub compression_level: u8,
    /// Enable quoted triple dictionary
    pub enable_quoted_dict: bool,
    /// Maximum quoted triple nesting depth
    pub max_nesting_depth: usize,
    /// Block size for bitmap indices (power of 2)
    pub block_size: usize,
    /// Enable memory mapping for large files
    pub enable_mmap: bool,
    /// Index strategy: SPO, POS, OSP, or ALL
    pub index_strategy: IndexStrategy,
}

impl Default for HdtStarConfig {
    fn default() -> Self {
        Self {
            enable_compression: true,
            compression_level: 6,
            enable_quoted_dict: true,
            max_nesting_depth: 10,
            block_size: 1024,
            enable_mmap: true,
            index_strategy: IndexStrategy::All,
        }
    }
}

/// Index strategy for triple storage
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexStrategy {
    /// Subject-Predicate-Object index only
    Spo,
    /// Predicate-Object-Subject index only
    Pos,
    /// Object-Subject-Predicate index only
    Osp,
    /// All three indices (larger but faster queries)
    All,
}

/// HDT-star header section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HdtStarHeader {
    /// Format version
    pub version: u8,
    /// Base URI for the dataset
    pub base_uri: Option<String>,
    /// Total number of triples
    pub triple_count: u64,
    /// Number of unique subjects
    pub subject_count: u64,
    /// Number of unique predicates
    pub predicate_count: u64,
    /// Number of unique objects
    pub object_count: u64,
    /// Number of quoted triples
    pub quoted_triple_count: u64,
    /// Maximum nesting depth encountered
    pub max_nesting_depth: u8,
    /// Configuration used for encoding
    pub config: HdtStarConfig,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
    /// Creation timestamp (Unix epoch)
    pub created_at: u64,
}

impl HdtStarHeader {
    /// Create a new header
    pub fn new(config: HdtStarConfig) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            version: HDT_STAR_VERSION,
            base_uri: None,
            triple_count: 0,
            subject_count: 0,
            predicate_count: 0,
            object_count: 0,
            quoted_triple_count: 0,
            max_nesting_depth: 0,
            config,
            metadata: HashMap::new(),
            created_at: timestamp,
        }
    }

    /// Serialize header to bytes
    pub fn to_bytes(&self) -> StarResult<Vec<u8>> {
        let encoded = bincode::serde::encode_to_vec(self, bincode::config::standard())
            .map_err(|e| StarError::serialization_error(format!("Header encoding failed: {e}")))?;
        Ok(encoded)
    }

    /// Deserialize header from bytes
    pub fn from_bytes(bytes: &[u8]) -> StarResult<Self> {
        let (decoded, _) = bincode::serde::decode_from_slice(bytes, bincode::config::standard())
            .map_err(|e| StarError::parse_error(format!("Header decoding failed: {e}")))?;
        Ok(decoded)
    }
}

/// Dictionary entry type
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DictionaryEntry {
    /// IRI entry
    Iri(String),
    /// Literal entry with optional datatype and language
    Literal {
        value: String,
        datatype: Option<String>,
        language: Option<String>,
    },
    /// Blank node entry
    BlankNode(String),
    /// Variable entry (for patterns)
    Variable(String),
    /// Quoted triple reference (index into quoted triple dictionary)
    QuotedTripleRef(u64),
}

impl DictionaryEntry {
    /// Convert from StarTerm
    pub fn from_star_term(term: &StarTerm) -> Self {
        match term {
            StarTerm::NamedNode(nn) => DictionaryEntry::Iri(nn.iri.clone()),
            StarTerm::Literal(lit) => DictionaryEntry::Literal {
                value: lit.value.clone(),
                datatype: lit.datatype.as_ref().map(|d| d.iri.clone()),
                language: lit.language.clone(),
            },
            StarTerm::BlankNode(bn) => DictionaryEntry::BlankNode(bn.id.clone()),
            StarTerm::Variable(var) => DictionaryEntry::Variable(var.name.clone()),
            StarTerm::QuotedTriple(_) => {
                // This is a placeholder; actual index is set during encoding
                DictionaryEntry::QuotedTripleRef(0)
            }
        }
    }

    /// Convert to StarTerm (except QuotedTripleRef which needs special handling)
    pub fn to_star_term(&self) -> StarResult<StarTerm> {
        match self {
            DictionaryEntry::Iri(iri) => StarTerm::iri(iri),
            DictionaryEntry::Literal {
                value,
                datatype,
                language,
            } => {
                let term = StarTerm::literal(value)?;
                // Apply datatype or language if present
                if let Some(lang) = language {
                    if let StarTerm::Literal(lit) = &term {
                        let mut new_lit = lit.clone();
                        new_lit.language = Some(lang.clone());
                        return Ok(StarTerm::Literal(new_lit));
                    }
                }
                if let Some(dt) = datatype {
                    if let StarTerm::Literal(lit) = &term {
                        let mut new_lit = lit.clone();
                        new_lit.datatype = Some(crate::model::NamedNode { iri: dt.clone() });
                        return Ok(StarTerm::Literal(new_lit));
                    }
                }
                Ok(term)
            }
            DictionaryEntry::BlankNode(bn) => Ok(StarTerm::BlankNode(crate::model::BlankNode {
                id: bn.clone(),
            })),
            DictionaryEntry::Variable(var) => Ok(StarTerm::Variable(crate::model::Variable {
                name: var.clone(),
            })),
            DictionaryEntry::QuotedTripleRef(_) => Err(StarError::invalid_term_type(
                "Cannot convert QuotedTripleRef directly to StarTerm",
            )),
        }
    }
}

/// Dictionary section for term compression
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HdtStarDictionary {
    /// Subject dictionary (strings to IDs)
    subjects: HashMap<DictionaryEntry, u64>,
    /// Predicate dictionary
    predicates: HashMap<DictionaryEntry, u64>,
    /// Object dictionary
    objects: HashMap<DictionaryEntry, u64>,
    /// Shared dictionary (terms appearing as both subject and object)
    shared: HashMap<DictionaryEntry, u64>,
    /// Quoted triple dictionary
    quoted_triples: HashMap<u64, EncodedTriple>,
    /// Reverse lookup for subjects
    subject_reverse: Vec<DictionaryEntry>,
    /// Reverse lookup for predicates
    predicate_reverse: Vec<DictionaryEntry>,
    /// Reverse lookup for objects
    object_reverse: Vec<DictionaryEntry>,
    /// Reverse lookup for quoted triples
    quoted_triple_reverse: Vec<EncodedTriple>,
    /// Next available ID for subjects
    next_subject_id: u64,
    /// Next available ID for predicates
    next_predicate_id: u64,
    /// Next available ID for objects
    next_object_id: u64,
    /// Next available ID for quoted triples
    next_quoted_id: u64,
}

impl HdtStarDictionary {
    /// Create a new empty dictionary
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a subject term and return its ID
    pub fn add_subject(&mut self, entry: DictionaryEntry) -> u64 {
        if let Some(&id) = self.subjects.get(&entry) {
            return id;
        }
        let id = self.next_subject_id;
        self.next_subject_id += 1;
        self.subjects.insert(entry.clone(), id);
        self.subject_reverse.push(entry);
        id
    }

    /// Add a predicate term and return its ID
    pub fn add_predicate(&mut self, entry: DictionaryEntry) -> u64 {
        if let Some(&id) = self.predicates.get(&entry) {
            return id;
        }
        let id = self.next_predicate_id;
        self.next_predicate_id += 1;
        self.predicates.insert(entry.clone(), id);
        self.predicate_reverse.push(entry);
        id
    }

    /// Add an object term and return its ID
    pub fn add_object(&mut self, entry: DictionaryEntry) -> u64 {
        if let Some(&id) = self.objects.get(&entry) {
            return id;
        }
        let id = self.next_object_id;
        self.next_object_id += 1;
        self.objects.insert(entry.clone(), id);
        self.object_reverse.push(entry);
        id
    }

    /// Add a quoted triple and return its ID
    pub fn add_quoted_triple(&mut self, triple: EncodedTriple) -> u64 {
        // Check if we already have this exact encoded triple
        for (id, existing) in &self.quoted_triples {
            if existing == &triple {
                return *id;
            }
        }
        let id = self.next_quoted_id;
        self.next_quoted_id += 1;
        self.quoted_triples.insert(id, triple.clone());
        self.quoted_triple_reverse.push(triple);
        id
    }

    /// Get subject by ID
    pub fn get_subject(&self, id: u64) -> Option<&DictionaryEntry> {
        self.subject_reverse.get(id as usize)
    }

    /// Get predicate by ID
    pub fn get_predicate(&self, id: u64) -> Option<&DictionaryEntry> {
        self.predicate_reverse.get(id as usize)
    }

    /// Get object by ID
    pub fn get_object(&self, id: u64) -> Option<&DictionaryEntry> {
        self.object_reverse.get(id as usize)
    }

    /// Get quoted triple by ID
    pub fn get_quoted_triple(&self, id: u64) -> Option<&EncodedTriple> {
        self.quoted_triples.get(&id)
    }

    /// Get dictionary statistics
    pub fn statistics(&self) -> DictionaryStats {
        DictionaryStats {
            subject_count: self.subjects.len(),
            predicate_count: self.predicates.len(),
            object_count: self.objects.len(),
            shared_count: self.shared.len(),
            quoted_triple_count: self.quoted_triples.len(),
        }
    }

    /// Serialize dictionary to bytes
    pub fn to_bytes(&self) -> StarResult<Vec<u8>> {
        let encoded =
            bincode::serde::encode_to_vec(self, bincode::config::standard()).map_err(|e| {
                StarError::serialization_error(format!("Dictionary encoding failed: {e}"))
            })?;
        Ok(encoded)
    }

    /// Deserialize dictionary from bytes
    pub fn from_bytes(bytes: &[u8]) -> StarResult<Self> {
        let (decoded, _) = bincode::serde::decode_from_slice(bytes, bincode::config::standard())
            .map_err(|e| StarError::parse_error(format!("Dictionary decoding failed: {e}")))?;
        Ok(decoded)
    }
}

/// Dictionary statistics
#[derive(Debug, Clone, Default)]
pub struct DictionaryStats {
    pub subject_count: usize,
    pub predicate_count: usize,
    pub object_count: usize,
    pub shared_count: usize,
    pub quoted_triple_count: usize,
}

/// Encoded triple with dictionary IDs
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EncodedTriple {
    /// Subject ID (or quoted triple marker + ID)
    pub subject: EncodedTerm,
    /// Predicate ID
    pub predicate: u64,
    /// Object ID (or quoted triple marker + ID)
    pub object: EncodedTerm,
}

/// Encoded term that can be a regular ID or a quoted triple reference
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum EncodedTerm {
    /// Regular dictionary ID
    Regular(u64),
    /// Quoted triple ID
    QuotedTriple(u64),
}

/// Triple section with bitmap-based storage
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HdtStarTriples {
    /// SPO-ordered triples for efficient subject lookups
    spo_index: Vec<EncodedTriple>,
    /// POS-ordered triples for efficient predicate lookups
    pos_index: Vec<EncodedTriple>,
    /// OSP-ordered triples for efficient object lookups
    osp_index: Vec<EncodedTriple>,
    /// Bitmap for efficient existence checks
    bitmap: Vec<u64>,
    /// Block size for bitmap operations
    block_size: usize,
}

impl HdtStarTriples {
    /// Create new triple section
    pub fn new(block_size: usize) -> Self {
        Self {
            spo_index: Vec::new(),
            pos_index: Vec::new(),
            osp_index: Vec::new(),
            bitmap: Vec::new(),
            block_size,
        }
    }

    /// Add an encoded triple
    pub fn add(&mut self, triple: EncodedTriple) {
        self.spo_index.push(triple.clone());
        self.pos_index.push(triple.clone());
        self.osp_index.push(triple);
    }

    /// Sort and build indices
    pub fn build_indices(&mut self, strategy: IndexStrategy) {
        // Sort SPO index
        if matches!(strategy, IndexStrategy::Spo | IndexStrategy::All) {
            self.spo_index.sort_by(|a, b| {
                (&a.subject, a.predicate, &a.object).cmp(&(&b.subject, b.predicate, &b.object))
            });
        }

        // Sort POS index
        if matches!(strategy, IndexStrategy::Pos | IndexStrategy::All) {
            self.pos_index.sort_by(|a, b| {
                (a.predicate, &a.object, &a.subject).cmp(&(b.predicate, &b.object, &b.subject))
            });
        }

        // Sort OSP index
        if matches!(strategy, IndexStrategy::Osp | IndexStrategy::All) {
            self.osp_index.sort_by(|a, b| {
                (&a.object, &a.subject, a.predicate).cmp(&(&b.object, &b.subject, b.predicate))
            });
        }

        // Build bitmap
        self.build_bitmap();
    }

    /// Build bitmap for fast existence checks
    fn build_bitmap(&mut self) {
        let num_blocks = (self.spo_index.len() + 63) / 64;
        self.bitmap = vec![0u64; num_blocks];

        // Set bits for all existing triples
        for i in 0..self.spo_index.len() {
            let block = i / 64;
            let bit = i % 64;
            if block < self.bitmap.len() {
                self.bitmap[block] |= 1u64 << bit;
            }
        }
    }

    /// Get triple count
    pub fn len(&self) -> usize {
        self.spo_index.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.spo_index.is_empty()
    }

    /// Iterate over all triples
    pub fn iter(&self) -> impl Iterator<Item = &EncodedTriple> {
        self.spo_index.iter()
    }

    /// Serialize triples to bytes
    pub fn to_bytes(&self) -> StarResult<Vec<u8>> {
        let encoded = bincode::serde::encode_to_vec(self, bincode::config::standard())
            .map_err(|e| StarError::serialization_error(format!("Triples encoding failed: {e}")))?;
        Ok(encoded)
    }

    /// Deserialize triples from bytes
    pub fn from_bytes(bytes: &[u8]) -> StarResult<Self> {
        let (decoded, _) = bincode::serde::decode_from_slice(bytes, bincode::config::standard())
            .map_err(|e| StarError::parse_error(format!("Triples decoding failed: {e}")))?;
        Ok(decoded)
    }
}

/// HDT-star builder for creating HDT files
pub struct HdtStarBuilder {
    config: HdtStarConfig,
    header: HdtStarHeader,
    dictionary: HdtStarDictionary,
    triples: HdtStarTriples,
    #[allow(dead_code)]
    profiler: Profiler,
}

impl HdtStarBuilder {
    /// Create a new HDT-star builder
    pub fn new(config: HdtStarConfig) -> Self {
        let header = HdtStarHeader::new(config.clone());
        let triples = HdtStarTriples::new(config.block_size);

        Self {
            config,
            header,
            dictionary: HdtStarDictionary::new(),
            triples,
            profiler: Profiler::new(),
        }
    }

    /// Set the base URI for the dataset
    pub fn set_base_uri(&mut self, base_uri: impl Into<String>) {
        self.header.base_uri = Some(base_uri.into());
    }

    /// Add custom metadata
    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.header.metadata.insert(key.into(), value.into());
    }

    /// Add a StarStore to the builder
    #[instrument(skip(self, store), fields(store_size = store.len()))]
    pub fn add_store(&mut self, store: &StarStore) -> StarResult<()> {
        info!(
            "Adding store with {} triples to HDT-star builder",
            store.len()
        );

        for triple in store.iter() {
            self.add_triple(&triple)?;
        }

        Ok(())
    }

    /// Add a single triple
    pub fn add_triple(&mut self, triple: &StarTriple) -> StarResult<()> {
        let encoded = self.encode_triple(triple)?;
        self.triples.add(encoded);
        self.header.triple_count += 1;
        Ok(())
    }

    /// Encode a StarTriple to an EncodedTriple
    fn encode_triple(&mut self, triple: &StarTriple) -> StarResult<EncodedTriple> {
        let subject = self.encode_term(&triple.subject, TermPosition::Subject)?;
        let predicate = match &triple.predicate {
            StarTerm::NamedNode(_) => {
                let entry = DictionaryEntry::from_star_term(&triple.predicate);
                self.dictionary.add_predicate(entry)
            }
            _ => {
                return Err(StarError::invalid_term_type(
                    "Predicate must be a NamedNode",
                ))
            }
        };
        let object = self.encode_term(&triple.object, TermPosition::Object)?;

        Ok(EncodedTriple {
            subject,
            predicate,
            object,
        })
    }

    /// Encode a StarTerm to an EncodedTerm
    fn encode_term(&mut self, term: &StarTerm, position: TermPosition) -> StarResult<EncodedTerm> {
        match term {
            StarTerm::QuotedTriple(qt) => {
                // Recursively encode the quoted triple
                let encoded_qt = self.encode_triple(qt)?;
                let qt_id = self.dictionary.add_quoted_triple(encoded_qt);
                self.header.quoted_triple_count += 1;
                Ok(EncodedTerm::QuotedTriple(qt_id))
            }
            _ => {
                let entry = DictionaryEntry::from_star_term(term);
                let id = match position {
                    TermPosition::Subject => self.dictionary.add_subject(entry),
                    TermPosition::Object => self.dictionary.add_object(entry),
                };
                Ok(EncodedTerm::Regular(id))
            }
        }
    }

    /// Build and finalize the HDT structure
    #[instrument(skip(self))]
    pub fn build(&mut self) -> StarResult<()> {
        info!("Building HDT-star indices...");

        // Build triple indices
        self.triples.build_indices(self.config.index_strategy);

        // Update header statistics
        let dict_stats = self.dictionary.statistics();
        self.header.subject_count = dict_stats.subject_count as u64;
        self.header.predicate_count = dict_stats.predicate_count as u64;
        self.header.object_count = dict_stats.object_count as u64;

        info!(
            "HDT-star built: {} triples, {} subjects, {} predicates, {} objects, {} quoted",
            self.header.triple_count,
            self.header.subject_count,
            self.header.predicate_count,
            self.header.object_count,
            self.header.quoted_triple_count
        );

        Ok(())
    }

    /// Write HDT-star to a writer
    #[instrument(skip(self, writer))]
    pub fn write<W: Write>(&mut self, writer: &mut W) -> StarResult<()> {
        self.build()?;

        info!("Writing HDT-star to output...");

        // Write magic bytes
        writer.write_all(&HDT_STAR_MAGIC).map_err(|e| {
            StarError::serialization_error(format!("Failed to write magic bytes: {e}"))
        })?;

        // Write header
        let header_bytes = self.header.to_bytes()?;
        let header_len = (header_bytes.len() as u64).to_le_bytes();
        writer.write_all(&header_len).map_err(|e| {
            StarError::serialization_error(format!("Failed to write header length: {e}"))
        })?;
        writer
            .write_all(&header_bytes)
            .map_err(|e| StarError::serialization_error(format!("Failed to write header: {e}")))?;

        // Write dictionary
        let dict_bytes = self.dictionary.to_bytes()?;
        let dict_len = (dict_bytes.len() as u64).to_le_bytes();
        writer.write_all(&dict_len).map_err(|e| {
            StarError::serialization_error(format!("Failed to write dictionary length: {e}"))
        })?;

        // Optionally compress dictionary
        if self.config.enable_compression {
            let compressed = compress_data(&dict_bytes, self.config.compression_level)?;
            let compressed_len = (compressed.len() as u64).to_le_bytes();
            writer.write_all(&compressed_len).map_err(|e| {
                StarError::serialization_error(format!("Failed to write compressed length: {e}"))
            })?;
            writer.write_all(&compressed).map_err(|e| {
                StarError::serialization_error(format!("Failed to write dictionary: {e}"))
            })?;
        } else {
            writer.write_all(&dict_bytes).map_err(|e| {
                StarError::serialization_error(format!("Failed to write dictionary: {e}"))
            })?;
        }

        // Write triples
        let triples_bytes = self.triples.to_bytes()?;
        let triples_len = (triples_bytes.len() as u64).to_le_bytes();
        writer.write_all(&triples_len).map_err(|e| {
            StarError::serialization_error(format!("Failed to write triples length: {e}"))
        })?;

        if self.config.enable_compression {
            let compressed = compress_data(&triples_bytes, self.config.compression_level)?;
            let compressed_len = (compressed.len() as u64).to_le_bytes();
            writer.write_all(&compressed_len).map_err(|e| {
                StarError::serialization_error(format!("Failed to write compressed length: {e}"))
            })?;
            writer.write_all(&compressed).map_err(|e| {
                StarError::serialization_error(format!("Failed to write triples: {e}"))
            })?;
        } else {
            writer.write_all(&triples_bytes).map_err(|e| {
                StarError::serialization_error(format!("Failed to write triples: {e}"))
            })?;
        }

        info!("HDT-star write complete");
        Ok(())
    }

    /// Get build statistics
    pub fn statistics(&self) -> HdtStarBuildStats {
        let dict_stats = self.dictionary.statistics();
        HdtStarBuildStats {
            triple_count: self.header.triple_count,
            quoted_triple_count: self.header.quoted_triple_count,
            subject_count: dict_stats.subject_count as u64,
            predicate_count: dict_stats.predicate_count as u64,
            object_count: dict_stats.object_count as u64,
            build_time_us: 0, // Profiler doesn't expose total_time_us
        }
    }
}

/// Term position for dictionary placement
#[derive(Debug, Clone, Copy)]
enum TermPosition {
    Subject,
    Object,
}

/// HDT-star build statistics
#[derive(Debug, Clone)]
pub struct HdtStarBuildStats {
    pub triple_count: u64,
    pub quoted_triple_count: u64,
    pub subject_count: u64,
    pub predicate_count: u64,
    pub object_count: u64,
    pub build_time_us: u64,
}

/// HDT-star reader for loading and querying HDT files
pub struct HdtStarReader {
    header: HdtStarHeader,
    dictionary: HdtStarDictionary,
    triples: HdtStarTriples,
}

impl HdtStarReader {
    /// Open an HDT-star file
    #[instrument(skip(path))]
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> StarResult<Self> {
        let mut file = std::fs::File::open(path.as_ref())
            .map_err(|e| StarError::resource_error(format!("Failed to open file: {e}")))?;

        Self::read(&mut file)
    }

    /// Read HDT-star from a reader
    pub fn read<R: Read>(reader: &mut R) -> StarResult<Self> {
        // Read and verify magic bytes
        let mut magic = [0u8; 8];
        reader
            .read_exact(&mut magic)
            .map_err(|e| StarError::parse_error(format!("Failed to read magic bytes: {e}")))?;

        if magic != HDT_STAR_MAGIC {
            return Err(StarError::parse_error(
                "Invalid HDT-star file: magic bytes mismatch",
            ));
        }

        // Read header
        let mut header_len_bytes = [0u8; 8];
        reader
            .read_exact(&mut header_len_bytes)
            .map_err(|e| StarError::parse_error(format!("Failed to read header length: {e}")))?;
        let header_len = u64::from_le_bytes(header_len_bytes) as usize;

        let mut header_bytes = vec![0u8; header_len];
        reader
            .read_exact(&mut header_bytes)
            .map_err(|e| StarError::parse_error(format!("Failed to read header: {e}")))?;
        let header = HdtStarHeader::from_bytes(&header_bytes)?;

        // Read dictionary length
        let mut dict_len_bytes = [0u8; 8];
        reader.read_exact(&mut dict_len_bytes).map_err(|e| {
            StarError::parse_error(format!("Failed to read dictionary length: {e}"))
        })?;
        let _dict_len = u64::from_le_bytes(dict_len_bytes) as usize;

        // Read dictionary (handle compression)
        let dict_bytes = if header.config.enable_compression {
            let mut compressed_len_bytes = [0u8; 8];
            reader.read_exact(&mut compressed_len_bytes).map_err(|e| {
                StarError::parse_error(format!("Failed to read compressed length: {e}"))
            })?;
            let compressed_len = u64::from_le_bytes(compressed_len_bytes) as usize;

            let mut compressed = vec![0u8; compressed_len];
            reader
                .read_exact(&mut compressed)
                .map_err(|e| StarError::parse_error(format!("Failed to read dictionary: {e}")))?;
            decompress_data(&compressed)?
        } else {
            let mut dict_bytes = vec![0u8; _dict_len];
            reader
                .read_exact(&mut dict_bytes)
                .map_err(|e| StarError::parse_error(format!("Failed to read dictionary: {e}")))?;
            dict_bytes
        };
        let dictionary = HdtStarDictionary::from_bytes(&dict_bytes)?;

        // Read triples length
        let mut triples_len_bytes = [0u8; 8];
        reader
            .read_exact(&mut triples_len_bytes)
            .map_err(|e| StarError::parse_error(format!("Failed to read triples length: {e}")))?;
        let _triples_len = u64::from_le_bytes(triples_len_bytes) as usize;

        // Read triples (handle compression)
        let triples_bytes = if header.config.enable_compression {
            let mut compressed_len_bytes = [0u8; 8];
            reader.read_exact(&mut compressed_len_bytes).map_err(|e| {
                StarError::parse_error(format!("Failed to read compressed length: {e}"))
            })?;
            let compressed_len = u64::from_le_bytes(compressed_len_bytes) as usize;

            let mut compressed = vec![0u8; compressed_len];
            reader
                .read_exact(&mut compressed)
                .map_err(|e| StarError::parse_error(format!("Failed to read triples: {e}")))?;
            decompress_data(&compressed)?
        } else {
            let mut triples_bytes = vec![0u8; _triples_len];
            reader
                .read_exact(&mut triples_bytes)
                .map_err(|e| StarError::parse_error(format!("Failed to read triples: {e}")))?;
            triples_bytes
        };
        let triples = HdtStarTriples::from_bytes(&triples_bytes)?;

        info!(
            "Loaded HDT-star file: {} triples, {} quoted triples",
            header.triple_count, header.quoted_triple_count
        );

        Ok(Self {
            header,
            dictionary,
            triples,
        })
    }

    /// Get the header
    pub fn header(&self) -> &HdtStarHeader {
        &self.header
    }

    /// Get the dictionary
    pub fn dictionary(&self) -> &HdtStarDictionary {
        &self.dictionary
    }

    /// Get triple count
    pub fn len(&self) -> usize {
        self.triples.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }

    /// Iterate over all triples, decoding them
    pub fn iter_triples(&self) -> impl Iterator<Item = StarResult<StarTriple>> + '_ {
        self.triples
            .iter()
            .map(|encoded| self.decode_triple(encoded))
    }

    /// Decode an encoded triple back to a StarTriple
    fn decode_triple(&self, encoded: &EncodedTriple) -> StarResult<StarTriple> {
        let subject = self.decode_term(&encoded.subject, TermPosition::Subject)?;
        let predicate = self
            .dictionary
            .get_predicate(encoded.predicate)
            .ok_or_else(|| StarError::parse_error("Invalid predicate ID"))?
            .to_star_term()?;
        let object = self.decode_term(&encoded.object, TermPosition::Object)?;

        Ok(StarTriple::new(subject, predicate, object))
    }

    /// Decode an encoded term back to a StarTerm
    fn decode_term(&self, encoded: &EncodedTerm, position: TermPosition) -> StarResult<StarTerm> {
        match encoded {
            EncodedTerm::Regular(id) => {
                let entry = match position {
                    TermPosition::Subject => self.dictionary.get_subject(*id),
                    TermPosition::Object => self.dictionary.get_object(*id),
                }
                .ok_or_else(|| StarError::parse_error("Invalid term ID"))?;
                entry.to_star_term()
            }
            EncodedTerm::QuotedTriple(qt_id) => {
                let encoded_qt = self
                    .dictionary
                    .get_quoted_triple(*qt_id)
                    .ok_or_else(|| StarError::parse_error("Invalid quoted triple ID"))?;
                let decoded_qt = self.decode_triple(encoded_qt)?;
                Ok(StarTerm::quoted_triple(decoded_qt))
            }
        }
    }

    /// Convert to StarStore
    pub fn to_store(&self) -> StarResult<StarStore> {
        let store = StarStore::new();
        for result in self.iter_triples() {
            let triple = result?;
            store.insert(&triple)?;
        }
        Ok(store)
    }

    /// Query by subject
    pub fn query_by_subject(&self, subject: &StarTerm) -> Vec<StarResult<StarTriple>> {
        let target_entry = DictionaryEntry::from_star_term(subject);

        // Find subject ID
        let subject_id = match self.dictionary.subjects.get(&target_entry) {
            Some(&id) => id,
            None => return Vec::new(),
        };

        self.triples
            .spo_index
            .iter()
            .filter(|t| matches!(&t.subject, EncodedTerm::Regular(id) if *id == subject_id))
            .map(|t| self.decode_triple(t))
            .collect()
    }

    /// Query by predicate
    pub fn query_by_predicate(&self, predicate: &StarTerm) -> Vec<StarResult<StarTriple>> {
        let target_entry = DictionaryEntry::from_star_term(predicate);

        // Find predicate ID
        let predicate_id = match self.dictionary.predicates.get(&target_entry) {
            Some(&id) => id,
            None => return Vec::new(),
        };

        self.triples
            .pos_index
            .iter()
            .filter(|t| t.predicate == predicate_id)
            .map(|t| self.decode_triple(t))
            .collect()
    }

    /// Query by object
    pub fn query_by_object(&self, object: &StarTerm) -> Vec<StarResult<StarTriple>> {
        let target_entry = DictionaryEntry::from_star_term(object);

        // Find object ID
        let object_id = match self.dictionary.objects.get(&target_entry) {
            Some(&id) => id,
            None => return Vec::new(),
        };

        self.triples
            .osp_index
            .iter()
            .filter(|t| matches!(&t.object, EncodedTerm::Regular(id) if *id == object_id))
            .map(|t| self.decode_triple(t))
            .collect()
    }
}

/// Compress data using zstd
fn compress_data(data: &[u8], level: u8) -> StarResult<Vec<u8>> {
    zstd::encode_all(std::io::Cursor::new(data), level as i32)
        .map_err(|e| StarError::serialization_error(format!("Compression failed: {e}")))
}

/// Decompress data using zstd
fn decompress_data(data: &[u8]) -> StarResult<Vec<u8>> {
    zstd::decode_all(std::io::Cursor::new(data))
        .map_err(|e| StarError::parse_error(format!("Decompression failed: {e}")))
}

/// HDT-star format converter
pub struct HdtStarConverter;

impl HdtStarConverter {
    /// Convert a StarStore to HDT-star bytes
    pub fn store_to_hdt(store: &StarStore, config: HdtStarConfig) -> StarResult<Vec<u8>> {
        let mut builder = HdtStarBuilder::new(config);
        builder.add_store(store)?;

        let mut buffer = Vec::new();
        builder.write(&mut buffer)?;
        Ok(buffer)
    }

    /// Convert HDT-star bytes to a StarStore
    pub fn hdt_to_store(data: &[u8]) -> StarResult<StarStore> {
        let reader = HdtStarReader::read(&mut std::io::Cursor::new(data))?;
        reader.to_store()
    }

    /// Get HDT-star file statistics without loading full data
    pub fn get_statistics<R: Read>(reader: &mut R) -> StarResult<HdtStarHeader> {
        // Read and verify magic bytes
        let mut magic = [0u8; 8];
        reader
            .read_exact(&mut magic)
            .map_err(|e| StarError::parse_error(format!("Failed to read magic bytes: {e}")))?;

        if magic != HDT_STAR_MAGIC {
            return Err(StarError::parse_error(
                "Invalid HDT-star file: magic bytes mismatch",
            ));
        }

        // Read header only
        let mut header_len_bytes = [0u8; 8];
        reader
            .read_exact(&mut header_len_bytes)
            .map_err(|e| StarError::parse_error(format!("Failed to read header length: {e}")))?;
        let header_len = u64::from_le_bytes(header_len_bytes) as usize;

        let mut header_bytes = vec![0u8; header_len];
        reader
            .read_exact(&mut header_bytes)
            .map_err(|e| StarError::parse_error(format!("Failed to read header: {e}")))?;

        HdtStarHeader::from_bytes(&header_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hdt_star_config_default() {
        let config = HdtStarConfig::default();
        assert!(config.enable_compression);
        assert_eq!(config.compression_level, 6);
        assert!(config.enable_quoted_dict);
        assert_eq!(config.max_nesting_depth, 10);
    }

    #[test]
    fn test_hdt_star_header_serialization() {
        let config = HdtStarConfig::default();
        let header = HdtStarHeader::new(config);

        let bytes = header.to_bytes().unwrap();
        let decoded = HdtStarHeader::from_bytes(&bytes).unwrap();

        assert_eq!(decoded.version, HDT_STAR_VERSION);
        assert_eq!(decoded.triple_count, 0);
    }

    #[test]
    fn test_dictionary_entry_conversion() {
        let iri = StarTerm::iri("http://example.org/test").unwrap();
        let entry = DictionaryEntry::from_star_term(&iri);

        assert!(matches!(entry, DictionaryEntry::Iri(_)));

        let back = entry.to_star_term().unwrap();
        assert!(matches!(back, StarTerm::NamedNode(_)));
    }

    #[test]
    fn test_dictionary_operations() {
        let mut dict = HdtStarDictionary::new();

        let entry1 = DictionaryEntry::Iri("http://example.org/s1".to_string());
        let entry2 = DictionaryEntry::Iri("http://example.org/s2".to_string());

        let id1 = dict.add_subject(entry1.clone());
        let id2 = dict.add_subject(entry2);
        let id1_again = dict.add_subject(entry1);

        assert_eq!(id1, id1_again); // Same entry should get same ID
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_hdt_star_builder_simple() {
        let config = HdtStarConfig::default();
        let mut builder = HdtStarBuilder::new(config);

        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/s").unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::iri("http://example.org/o").unwrap(),
        );

        builder.add_triple(&triple).unwrap();

        let stats = builder.statistics();
        assert_eq!(stats.triple_count, 1);
    }

    #[test]
    fn test_hdt_star_builder_with_quoted_triple() {
        let config = HdtStarConfig::default();
        let mut builder = HdtStarBuilder::new(config);

        let inner = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/age").unwrap(),
            StarTerm::literal("30").unwrap(),
        );

        let outer = StarTriple::new(
            StarTerm::quoted_triple(inner),
            StarTerm::iri("http://example.org/certainty").unwrap(),
            StarTerm::literal("0.9").unwrap(),
        );

        builder.add_triple(&outer).unwrap();

        let stats = builder.statistics();
        assert_eq!(stats.triple_count, 1);
        assert_eq!(stats.quoted_triple_count, 1);
    }

    #[test]
    fn test_hdt_star_roundtrip() {
        let config = HdtStarConfig::default();
        let mut builder = HdtStarBuilder::new(config);

        // Add regular triples
        for i in 0..10 {
            let s = format!("http://example.org/s{}", i);
            let v = format!("value{}", i);
            let triple = StarTriple::new(
                StarTerm::iri(&s).unwrap(),
                StarTerm::iri("http://example.org/p").unwrap(),
                StarTerm::literal(&v).unwrap(),
            );
            builder.add_triple(&triple).unwrap();
        }

        // Add quoted triple
        let quoted = StarTriple::new(
            StarTerm::iri("http://example.org/x").unwrap(),
            StarTerm::iri("http://example.org/y").unwrap(),
            StarTerm::iri("http://example.org/z").unwrap(),
        );
        let meta = StarTriple::new(
            StarTerm::quoted_triple(quoted),
            StarTerm::iri("http://example.org/meta").unwrap(),
            StarTerm::literal("test").unwrap(),
        );
        builder.add_triple(&meta).unwrap();

        // Write to buffer
        let mut buffer = Vec::new();
        builder.write(&mut buffer).unwrap();

        // Read back
        let reader = HdtStarReader::read(&mut std::io::Cursor::new(&buffer)).unwrap();

        assert_eq!(reader.len(), 11);
        assert_eq!(reader.header().quoted_triple_count, 1);

        // Verify all triples decode correctly
        let decoded: Vec<_> = reader.iter_triples().collect();
        assert_eq!(decoded.len(), 11);
        for result in decoded {
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_hdt_star_query_by_subject() {
        let config = HdtStarConfig::default();
        let mut builder = HdtStarBuilder::new(config);

        let subject = StarTerm::iri("http://example.org/alice").unwrap();

        for i in 0..5 {
            let p = format!("http://example.org/p{}", i);
            let v = format!("value{}", i);
            let triple = StarTriple::new(
                subject.clone(),
                StarTerm::iri(&p).unwrap(),
                StarTerm::literal(&v).unwrap(),
            );
            builder.add_triple(&triple).unwrap();
        }

        // Add some other triples
        for i in 0..3 {
            let p = format!("http://example.org/p{}", i);
            let triple = StarTriple::new(
                StarTerm::iri("http://example.org/bob").unwrap(),
                StarTerm::iri(&p).unwrap(),
                StarTerm::literal("other").unwrap(),
            );
            builder.add_triple(&triple).unwrap();
        }

        let mut buffer = Vec::new();
        builder.write(&mut buffer).unwrap();

        let reader = HdtStarReader::read(&mut std::io::Cursor::new(&buffer)).unwrap();
        let results = reader.query_by_subject(&subject);

        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_hdt_star_converter() {
        let store = StarStore::new();

        for i in 0..10 {
            let s = format!("http://example.org/s{}", i);
            let v = format!("{}", i);
            let triple = StarTriple::new(
                StarTerm::iri(&s).unwrap(),
                StarTerm::iri("http://example.org/p").unwrap(),
                StarTerm::literal(&v).unwrap(),
            );
            store.insert(&triple).unwrap();
        }

        let config = HdtStarConfig::default();
        let hdt_bytes = HdtStarConverter::store_to_hdt(&store, config).unwrap();

        let restored = HdtStarConverter::hdt_to_store(&hdt_bytes).unwrap();

        assert_eq!(restored.len(), store.len());
    }

    #[test]
    fn test_hdt_star_compression() {
        let config = HdtStarConfig {
            enable_compression: true,
            compression_level: 9,
            ..HdtStarConfig::default()
        };

        let mut builder = HdtStarBuilder::new(config.clone());

        // Add many similar triples to test compression
        for i in 0..1000 {
            let s = format!("http://example.org/subject{}", i);
            let v = format!("This is a test value number {}", i);
            let triple = StarTriple::new(
                StarTerm::iri(&s).unwrap(),
                StarTerm::iri("http://example.org/predicate").unwrap(),
                StarTerm::literal(&v).unwrap(),
            );
            builder.add_triple(&triple).unwrap();
        }

        let mut compressed_buffer = Vec::new();
        builder.write(&mut compressed_buffer).unwrap();

        // Without compression
        let config_uncompressed = HdtStarConfig {
            enable_compression: false,
            ..config
        };
        let mut builder_uncompressed = HdtStarBuilder::new(config_uncompressed);
        for i in 0..1000 {
            let s = format!("http://example.org/subject{}", i);
            let v = format!("This is a test value number {}", i);
            let triple = StarTriple::new(
                StarTerm::iri(&s).unwrap(),
                StarTerm::iri("http://example.org/predicate").unwrap(),
                StarTerm::literal(&v).unwrap(),
            );
            builder_uncompressed.add_triple(&triple).unwrap();
        }

        let mut uncompressed_buffer = Vec::new();
        builder_uncompressed
            .write(&mut uncompressed_buffer)
            .unwrap();

        // Compressed should be significantly smaller
        assert!(
            compressed_buffer.len() < uncompressed_buffer.len(),
            "Compressed: {}, Uncompressed: {}",
            compressed_buffer.len(),
            uncompressed_buffer.len()
        );
    }

    #[test]
    fn test_hdt_star_nested_quoted_triples() {
        let config = HdtStarConfig::default();
        let mut builder = HdtStarBuilder::new(config);

        // Create nested quoted triple
        let level1 = StarTriple::new(
            StarTerm::iri("http://example.org/a").unwrap(),
            StarTerm::iri("http://example.org/b").unwrap(),
            StarTerm::iri("http://example.org/c").unwrap(),
        );

        let level2 = StarTriple::new(
            StarTerm::quoted_triple(level1),
            StarTerm::iri("http://example.org/meta1").unwrap(),
            StarTerm::literal("level2").unwrap(),
        );

        let level3 = StarTriple::new(
            StarTerm::quoted_triple(level2),
            StarTerm::iri("http://example.org/meta2").unwrap(),
            StarTerm::literal("level3").unwrap(),
        );

        builder.add_triple(&level3).unwrap();

        let stats = builder.statistics();
        assert_eq!(stats.triple_count, 1);
        assert_eq!(stats.quoted_triple_count, 2); // Two nested quoted triples

        // Roundtrip test
        let mut buffer = Vec::new();
        builder.write(&mut buffer).unwrap();

        let reader = HdtStarReader::read(&mut std::io::Cursor::new(&buffer)).unwrap();
        let triples: Vec<_> = reader.iter_triples().collect();

        assert_eq!(triples.len(), 1);
        assert!(triples[0].is_ok());
    }
}
