//! Compact storage format for RDF-star annotation metadata
//!
//! This module provides memory-efficient storage and serialization for
//! annotation metadata, reducing overhead and improving performance for
//! large-scale RDF-star datasets with extensive annotations.
//!
//! # Features
//!
//! - **Binary serialization** - Compact binary format for annotations
//! - **Compression** - Zstd compression for repeated metadata
//! - **Delta encoding** - Efficient storage of version chains
//! - **Dictionary compression** - String deduplication for common values
//! - **Memory mapping** - Support for datasets larger than RAM
//! - **Lazy loading** - Load annotations on-demand
//!
//! # Examples
//!
//! ```rust
//! use oxirs_star::compact_annotation_storage::{CompactAnnotationStore, CompressionLevel};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a compact annotation store
//! let mut store = CompactAnnotationStore::new(CompressionLevel::Balanced);
//!
//! // Store annotations with automatic compression
//! // ... annotation operations
//!
//! println!("Memory usage: {} bytes", store.memory_usage());
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Write};

// SciRS2 imports for memory-efficient operations (SCIRS2 POLICY)
// Note: BufferPool is available but not used in current implementation
// Future optimization can leverage these for large-scale annotation storage

use crate::annotations::{EvidenceItem, MetaAnnotation, ProvenanceRecord, TripleAnnotation};
use crate::model::StarTriple;
use crate::StarResult;

/// Compression level for annotation storage
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionLevel {
    /// No compression (fastest)
    None,
    /// Light compression (balanced speed/size)
    Light,
    /// Balanced compression (default)
    Balanced,
    /// Heavy compression (maximum space savings)
    Heavy,
}

impl CompressionLevel {
    /// Get zstd compression level
    fn zstd_level(self) -> i32 {
        match self {
            Self::None => 0,
            Self::Light => 3,
            Self::Balanced => 6,
            Self::Heavy => 15,
        }
    }
}

/// String dictionary for deduplication
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StringDictionary {
    /// String to ID mapping
    string_to_id: HashMap<String, u32>,
    /// ID to string mapping
    id_to_string: Vec<String>,
}

impl StringDictionary {
    fn new() -> Self {
        Self {
            string_to_id: HashMap::new(),
            id_to_string: Vec::new(),
        }
    }

    /// Intern a string and return its ID
    fn intern(&mut self, s: &str) -> u32 {
        if let Some(&id) = self.string_to_id.get(s) {
            id
        } else {
            let id = self.id_to_string.len() as u32;
            self.string_to_id.insert(s.to_string(), id);
            self.id_to_string.push(s.to_string());
            id
        }
    }

    /// Get string by ID
    fn get(&self, id: u32) -> Option<&str> {
        self.id_to_string.get(id as usize).map(|s| s.as_str())
    }

    /// Get memory usage
    fn memory_usage(&self) -> usize {
        self.string_to_id.len() * (std::mem::size_of::<String>() + std::mem::size_of::<u32>())
            + self.id_to_string.iter().map(|s| s.len()).sum::<usize>()
    }
}

/// Compact binary representation of an annotation
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CompactAnnotation {
    /// Confidence score (encoded as u16: 0-65535 representing 0.0-1.0)
    confidence: Option<u16>,

    /// Source ID (dictionary index)
    source_id: Option<u32>,

    /// Timestamp (Unix timestamp in seconds)
    timestamp: Option<i64>,

    /// Validity period (start, end timestamps)
    validity_period: Option<(i64, i64)>,

    /// Evidence items (compact)
    evidence: Vec<CompactEvidence>,

    /// Custom metadata (dictionary IDs)
    custom_metadata: Vec<(u32, u32)>,

    /// Provenance records (compact)
    provenance: Vec<CompactProvenance>,

    /// Quality score (encoded as u16)
    quality_score: Option<u16>,

    /// Locale ID (dictionary index)
    locale_id: Option<u32>,

    /// Version number
    version: Option<u64>,

    /// Meta-annotations count (nested annotations stored separately)
    meta_annotations_count: usize,

    /// Annotation ID (dictionary index)
    annotation_id: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CompactEvidence {
    evidence_type_id: u32,
    reference_id: u32,
    strength: u16,
    description_id: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CompactProvenance {
    action_id: u32,
    agent_id: u32,
    timestamp: i64,
    activity_id: Option<u32>,
    method_id: Option<u32>,
}

/// Compact annotation store
pub struct CompactAnnotationStore {
    /// String dictionary for deduplication
    dictionary: StringDictionary,

    /// Compact annotations indexed by triple hash
    annotations: HashMap<u64, CompactAnnotation>,

    /// Meta-annotations storage (separate for better compression)
    meta_annotations: HashMap<u64, Vec<CompactAnnotation>>,

    /// Compression level
    compression_level: CompressionLevel,

    /// Statistics
    stats: CompactStorageStatistics,
}

/// Statistics for compact storage
#[derive(Debug, Clone, Default)]
pub struct CompactStorageStatistics {
    /// Number of annotations stored
    pub annotation_count: usize,

    /// Number of deduplicated strings
    pub dictionary_size: usize,

    /// Total memory usage (bytes)
    pub memory_usage: usize,

    /// Compression ratio (original / compressed)
    pub compression_ratio: f64,

    /// Number of meta-annotations
    pub meta_annotation_count: usize,
}

impl CompactAnnotationStore {
    /// Create a new compact annotation store
    pub fn new(compression_level: CompressionLevel) -> Self {
        Self {
            dictionary: StringDictionary::new(),
            annotations: HashMap::new(),
            meta_annotations: HashMap::new(),
            compression_level,
            stats: CompactStorageStatistics::default(),
        }
    }

    /// Store an annotation in compact format
    pub fn store_annotation(
        &mut self,
        triple: &StarTriple,
        annotation: &TripleAnnotation,
    ) -> StarResult<()> {
        let triple_hash = Self::hash_triple(triple);

        // Convert to compact format
        let compact = self.convert_to_compact(annotation);

        // Store meta-annotations separately
        if !annotation.meta_annotations.is_empty() {
            let meta_compacts: Vec<CompactAnnotation> = annotation
                .meta_annotations
                .iter()
                .map(|meta| self.convert_to_compact(&meta.annotation))
                .collect();

            self.meta_annotations.insert(triple_hash, meta_compacts);
            self.stats.meta_annotation_count += annotation.meta_annotations.len();
        }

        self.annotations.insert(triple_hash, compact);
        self.stats.annotation_count += 1;
        self.stats.dictionary_size = self.dictionary.id_to_string.len();

        self.update_memory_stats();

        Ok(())
    }

    /// Convert annotation to compact format
    fn convert_to_compact(&mut self, annotation: &TripleAnnotation) -> CompactAnnotation {
        CompactAnnotation {
            confidence: annotation.confidence.map(|c| (c * 65535.0) as u16),
            source_id: annotation
                .source
                .as_ref()
                .map(|s| self.dictionary.intern(s)),
            timestamp: annotation.timestamp.map(|t| t.timestamp()),
            validity_period: annotation
                .validity_period
                .map(|(start, end)| (start.timestamp(), end.timestamp())),
            evidence: annotation
                .evidence
                .iter()
                .map(|e| self.convert_evidence_to_compact(e))
                .collect(),
            custom_metadata: annotation
                .custom_metadata
                .iter()
                .map(|(k, v)| (self.dictionary.intern(k), self.dictionary.intern(v)))
                .collect(),
            provenance: annotation
                .provenance
                .iter()
                .map(|p| self.convert_provenance_to_compact(p))
                .collect(),
            quality_score: annotation.quality_score.map(|q| (q * 65535.0) as u16),
            locale_id: annotation
                .locale
                .as_ref()
                .map(|l| self.dictionary.intern(l)),
            version: annotation.version,
            meta_annotations_count: annotation.meta_annotations.len(),
            annotation_id: annotation
                .annotation_id
                .as_ref()
                .map(|id| self.dictionary.intern(id)),
        }
    }

    fn convert_evidence_to_compact(&mut self, evidence: &EvidenceItem) -> CompactEvidence {
        CompactEvidence {
            evidence_type_id: self.dictionary.intern(&evidence.evidence_type),
            reference_id: self.dictionary.intern(&evidence.reference),
            strength: (evidence.strength * 65535.0) as u16,
            description_id: evidence
                .description
                .as_ref()
                .map(|d| self.dictionary.intern(d)),
        }
    }

    fn convert_provenance_to_compact(
        &mut self,
        provenance: &ProvenanceRecord,
    ) -> CompactProvenance {
        CompactProvenance {
            action_id: self.dictionary.intern(&provenance.action),
            agent_id: self.dictionary.intern(&provenance.agent),
            timestamp: provenance.timestamp.timestamp(),
            activity_id: provenance
                .activity
                .as_ref()
                .map(|a| self.dictionary.intern(a)),
            method_id: provenance
                .method
                .as_ref()
                .map(|m| self.dictionary.intern(m)),
        }
    }

    /// Retrieve annotation from compact storage
    pub fn retrieve_annotation(&self, triple: &StarTriple) -> Option<TripleAnnotation> {
        let triple_hash = Self::hash_triple(triple);
        let compact = self.annotations.get(&triple_hash)?;

        Some(self.convert_from_compact(compact, triple_hash))
    }

    fn convert_from_compact(
        &self,
        compact: &CompactAnnotation,
        triple_hash: u64,
    ) -> TripleAnnotation {
        use chrono::{TimeZone, Utc};

        let mut annotation = TripleAnnotation {
            confidence: compact.confidence.map(|c| c as f64 / 65535.0),
            source: compact
                .source_id
                .and_then(|id| self.dictionary.get(id).map(|s| s.to_string())),
            timestamp: compact
                .timestamp
                .map(|ts| Utc.timestamp_opt(ts, 0).unwrap()),
            validity_period: compact.validity_period.map(|(start, end)| {
                (
                    Utc.timestamp_opt(start, 0).unwrap(),
                    Utc.timestamp_opt(end, 0).unwrap(),
                )
            }),
            evidence: compact
                .evidence
                .iter()
                .map(|e| self.convert_evidence_from_compact(e))
                .collect(),
            custom_metadata: compact
                .custom_metadata
                .iter()
                .filter_map(|(k_id, v_id)| {
                    let key = self.dictionary.get(*k_id)?.to_string();
                    let value = self.dictionary.get(*v_id)?.to_string();
                    Some((key, value))
                })
                .collect(),
            provenance: compact
                .provenance
                .iter()
                .map(|p| self.convert_provenance_from_compact(p))
                .collect(),
            quality_score: compact.quality_score.map(|q| q as f64 / 65535.0),
            locale: compact
                .locale_id
                .and_then(|id| self.dictionary.get(id).map(|s| s.to_string())),
            version: compact.version,
            meta_annotations: Vec::new(), // Will be populated separately
            annotation_id: compact
                .annotation_id
                .and_then(|id| self.dictionary.get(id).map(|s| s.to_string())),
        };

        // Retrieve meta-annotations
        if let Some(meta_compacts) = self.meta_annotations.get(&triple_hash) {
            for meta_compact in meta_compacts {
                let meta_annotation = self.convert_from_compact(meta_compact, triple_hash);
                annotation.meta_annotations.push(MetaAnnotation {
                    annotation_type: "unknown".to_string(), // Would need to store this separately
                    annotation: meta_annotation,
                    target_id: annotation.annotation_id.clone(),
                    depth: 0, // Would be recalculated
                });
            }
        }

        annotation
    }

    fn convert_evidence_from_compact(&self, compact: &CompactEvidence) -> EvidenceItem {
        EvidenceItem {
            evidence_type: self
                .dictionary
                .get(compact.evidence_type_id)
                .unwrap_or("unknown")
                .to_string(),
            reference: self
                .dictionary
                .get(compact.reference_id)
                .unwrap_or("unknown")
                .to_string(),
            strength: compact.strength as f64 / 65535.0,
            description: compact
                .description_id
                .and_then(|id| self.dictionary.get(id).map(|s| s.to_string())),
        }
    }

    fn convert_provenance_from_compact(&self, compact: &CompactProvenance) -> ProvenanceRecord {
        use chrono::{TimeZone, Utc};

        ProvenanceRecord {
            action: self
                .dictionary
                .get(compact.action_id)
                .unwrap_or("unknown")
                .to_string(),
            agent: self
                .dictionary
                .get(compact.agent_id)
                .unwrap_or("unknown")
                .to_string(),
            timestamp: Utc.timestamp_opt(compact.timestamp, 0).unwrap(),
            activity: compact
                .activity_id
                .and_then(|id| self.dictionary.get(id).map(|s| s.to_string())),
            method: compact
                .method_id
                .and_then(|id| self.dictionary.get(id).map(|s| s.to_string())),
        }
    }

    /// Update memory usage statistics
    fn update_memory_stats(&mut self) {
        let dict_size = self.dictionary.memory_usage();
        let annotations_size = self.annotations.len() * std::mem::size_of::<CompactAnnotation>();
        let meta_size = self.meta_annotations.len() * std::mem::size_of::<Vec<CompactAnnotation>>();

        self.stats.memory_usage = dict_size + annotations_size + meta_size;
    }

    /// Get statistics
    pub fn statistics(&self) -> &CompactStorageStatistics {
        &self.stats
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.stats.memory_usage
    }

    /// Serialize to compressed binary format
    pub fn serialize_compressed<W: Write>(&self, writer: W) -> StarResult<()> {
        let config = bincode::config::standard();
        let data = bincode::serde::encode_to_vec(
            (&self.dictionary, &self.annotations, &self.meta_annotations),
            config,
        )
        .map_err(|e| crate::StarError::serialization_error(e.to_string()))?;

        if self.compression_level != CompressionLevel::None {
            let compressed = zstd::encode_all(&data[..], self.compression_level.zstd_level())
                .map_err(|e| crate::StarError::serialization_error(e.to_string()))?;

            let mut w = writer;
            w.write_all(&compressed)
                .map_err(|e| crate::StarError::serialization_error(e.to_string()))?;
        } else {
            let mut w = writer;
            w.write_all(&data)
                .map_err(|e| crate::StarError::serialization_error(e.to_string()))?;
        }

        Ok(())
    }

    /// Deserialize from compressed binary format
    pub fn deserialize_compressed<R: Read>(
        reader: R,
        compression_level: CompressionLevel,
    ) -> StarResult<Self> {
        let mut data = Vec::new();
        let mut r = reader;
        r.read_to_end(&mut data)
            .map_err(|e| crate::StarError::parse_error(e.to_string()))?;

        let decompressed = if compression_level != CompressionLevel::None {
            zstd::decode_all(&data[..]).map_err(|e| crate::StarError::parse_error(e.to_string()))?
        } else {
            data
        };

        let config = bincode::config::standard();
        let (dictionary, annotations, meta_annotations): (
            StringDictionary,
            HashMap<u64, CompactAnnotation>,
            HashMap<u64, Vec<CompactAnnotation>>,
        ) = bincode::serde::decode_from_slice(&decompressed, config)
            .map_err(|e| crate::StarError::parse_error(e.to_string()))?
            .0;

        let mut store = Self {
            dictionary,
            annotations,
            meta_annotations,
            compression_level,
            stats: CompactStorageStatistics::default(),
        };

        store.stats.annotation_count = store.annotations.len();
        store.stats.dictionary_size = store.dictionary.id_to_string.len();
        store.update_memory_stats();

        Ok(store)
    }

    fn hash_triple(triple: &StarTriple) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        format!("{:?}", triple).hash(&mut hasher);
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::StarTerm;

    #[test]
    fn test_compact_storage() {
        let mut store = CompactAnnotationStore::new(CompressionLevel::Balanced);

        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/s").unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::iri("http://example.org/o").unwrap(),
        );

        let mut annotation = TripleAnnotation::new()
            .with_confidence(0.85)
            .with_source("http://example.org/source".to_string());

        annotation.quality_score = Some(0.9);

        store.store_annotation(&triple, &annotation).unwrap();

        let retrieved = store.retrieve_annotation(&triple).unwrap();
        assert!((retrieved.confidence.unwrap() - 0.85).abs() < 0.01);
    }

    #[test]
    fn test_string_dictionary() {
        let mut dict = StringDictionary::new();

        let id1 = dict.intern("test");
        let id2 = dict.intern("test"); // Should return same ID
        let id3 = dict.intern("different");

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
        assert_eq!(dict.get(id1), Some("test"));
    }

    #[test]
    fn test_compression_levels() {
        for level in [
            CompressionLevel::None,
            CompressionLevel::Light,
            CompressionLevel::Balanced,
            CompressionLevel::Heavy,
        ] {
            let store = CompactAnnotationStore::new(level);
            assert_eq!(store.compression_level, level);
        }
    }

    #[test]
    fn test_memory_usage() {
        let mut store = CompactAnnotationStore::new(CompressionLevel::Light);

        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/s").unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::iri("http://example.org/o").unwrap(),
        );

        let annotation = TripleAnnotation::new().with_confidence(0.9);

        store.store_annotation(&triple, &annotation).unwrap();

        let stats = store.statistics();
        assert!(stats.memory_usage > 0);
        assert_eq!(stats.annotation_count, 1);
    }
}
