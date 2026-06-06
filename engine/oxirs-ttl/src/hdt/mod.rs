//! # HDT Binary RDF Format Reader
//!
//! An iterator-based read-only parser for the **HDT 1.0** binary RDF format
//! as described by Martinez-Prieto et al.
//!
//! HDT (Header-Dictionary-Triples) stores large RDF datasets in three logical
//! sections:
//!
//! 1. **Header** — RDF metadata (triple count, subject/predicate/object counts,
//!    format info).
//! 2. **Dictionary** — four string sections (shared, subjects, predicates,
//!    objects) mapping integer IDs to RDF term strings.
//! 3. **Triples** — adjacency-list representation using integer IDs.
//!
//! # Quick Start
//!
//! ```rust
//! # use oxirs_ttl::hdt::{HdtDictionary, HdtHeader};
//! // Build a dictionary and header manually (useful in tests).
//! let mut dict = HdtDictionary::new();
//! dict.shared.push("<http://example.org/Alice>".to_owned());
//! dict.predicates.push("<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>".to_owned());
//! dict.objects.push("<http://schema.org/Person>".to_owned());
//!
//! assert_eq!(dict.lookup_subject(1), Some("<http://example.org/Alice>"));
//! assert_eq!(dict.lookup_predicate(1), Some("<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"));
//! assert_eq!(dict.lookup_object(2), Some("<http://schema.org/Person>"));
//! ```

pub mod dictionary;
pub mod format;
pub mod triples;

#[cfg(test)]
mod tests;

// Public re-exports from sub-modules
pub use dictionary::{parse_plain_dictionary, DictionarySection, HdtDictionary};
pub use format::{compute_crc16, compute_crc32, read_vbyte, read_vbyte_slice, write_vbyte};
pub use triples::HdtTriplesSection;

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

// ---------------------------------------------------------------------------
// HdtError
// ---------------------------------------------------------------------------

/// Errors that can arise while parsing an HDT file.
#[derive(Debug, thiserror::Error)]
pub enum HdtError {
    /// Underlying I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The HDT magic bytes were not found at the start of the file.
    #[error("invalid HDT magic bytes: found {:?}", got)]
    InvalidMagic {
        /// The bytes that were found instead of the expected magic.
        got: Vec<u8>,
    },

    /// A section header was malformed or absent.
    #[error("invalid HDT section: {name}")]
    InvalidSection {
        /// Human-readable description of the problem.
        name: String,
    },

    /// A dictionary ID could not be decoded to a string.
    #[error("dictionary decode error for id {id}")]
    DictionaryDecodeError {
        /// The ID that could not be resolved.
        id: u64,
    },

    /// A triples section entry could not be decoded.
    #[error("triple decode error: {msg}")]
    TripleDecodeError {
        /// Human-readable description of the problem.
        msg: String,
    },

    /// The HDT file uses a version not supported by this implementation.
    #[error("unsupported HDT version: {version}")]
    UnsupportedVersion {
        /// The version byte found in the file.
        version: u8,
    },
}

// ---------------------------------------------------------------------------
// HdtHeader
// ---------------------------------------------------------------------------

/// Parsed HDT header metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HdtHeader {
    /// Total number of triples in the dataset.
    pub triples_count: u64,
    /// Number of distinct subjects (including shared SO).
    pub subjects_count: u64,
    /// Number of distinct predicates.
    pub predicates_count: u64,
    /// Number of distinct objects (including shared SO).
    pub objects_count: u64,
    /// Number of shared subject/object nodes.
    pub shared_count: u64,
    /// Format string from the HDT metadata (e.g. `"hdt/plain"`).
    pub format: String,
    /// Base URI declared in the HDT header, if any.
    pub base_uri: Option<String>,
}

impl Default for HdtHeader {
    fn default() -> Self {
        HdtHeader {
            triples_count: 0,
            subjects_count: 0,
            predicates_count: 0,
            objects_count: 0,
            shared_count: 0,
            format: "hdt/plain".to_owned(),
            base_uri: None,
        }
    }
}

// ---------------------------------------------------------------------------
// HdtStats
// ---------------------------------------------------------------------------

/// Statistics about an HDT dataset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HdtStats {
    /// Total number of triples.
    pub triple_count: u64,
    /// Number of distinct subjects (shared + subject-only).
    pub distinct_subjects: u64,
    /// Number of distinct predicates.
    pub distinct_predicates: u64,
    /// Number of distinct objects (shared + object-only).
    pub distinct_objects: u64,
    /// Number of terms shared between subject and object positions.
    pub shared_so_count: u64,
}

// ---------------------------------------------------------------------------
// HdtTriple (resolved strings)
// ---------------------------------------------------------------------------

/// A resolved RDF triple with string subject, predicate, and object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HdtTriple {
    /// Subject IRI or blank-node label.
    pub subject: String,
    /// Predicate IRI.
    pub predicate: String,
    /// Object IRI or literal.
    pub object: String,
}

// ---------------------------------------------------------------------------
// HdtReader (bytes-based non-generic reader for backward compatibility)
// ---------------------------------------------------------------------------

/// Read-only iterator-based HDT 1.0 file parser.
///
/// Wraps either a file on disk or an in-memory byte buffer.  After
/// construction the header and dictionary are already parsed; the triple
/// iterator performs lazy resolution against the dictionary.
#[derive(Debug)]
pub struct HdtReader {
    header: HdtHeader,
    dictionary: HdtDictionary,
    triples_section: HdtTriplesSection,
}

/// Expected magic bytes at the beginning of every HDT 1.0 file (simplified).
const HDT_MAGIC_SIMPLE: &[u8] = b"$HDT\x01";

impl HdtReader {
    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    /// Open and parse an HDT file from the filesystem.
    ///
    /// # Errors
    /// Returns `HdtError::Io` for I/O failures and format-specific errors
    /// for malformed files.
    pub fn open(path: &Path) -> Result<Self, HdtError> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut data = Vec::new();
        reader.read_to_end(&mut data)?;
        Self::parse_bytes(&data)
    }

    /// Parse an HDT file from an in-memory byte vector.
    ///
    /// Convenient for tests and streaming scenarios where the caller has
    /// already loaded the bytes.
    ///
    /// # Errors
    /// Returns format-specific errors for malformed data.
    pub fn from_bytes(data: Vec<u8>) -> Result<Self, HdtError> {
        Self::parse_bytes(&data)
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Return a reference to the parsed HDT header metadata.
    pub fn header(&self) -> &HdtHeader {
        &self.header
    }

    /// Return the total number of triples.
    pub fn triple_count(&self) -> u64 {
        self.header.triples_count
    }

    /// Return statistics about this HDT dataset.
    pub fn stats(&self) -> HdtStats {
        HdtStats {
            triple_count: self.header.triples_count,
            distinct_subjects: self.header.subjects_count,
            distinct_predicates: self.header.predicates_count,
            distinct_objects: self.header.objects_count,
            shared_so_count: self.header.shared_count,
        }
    }

    /// Return a lazy iterator over all resolved triples.
    ///
    /// Each element is `Result<HdtTriple, HdtError>` where the error variant
    /// is `DictionaryDecodeError` if an ID cannot be resolved.
    pub fn triples(&self) -> impl Iterator<Item = Result<HdtTriple, HdtError>> + '_ {
        self.triples_section.iter_ids().map(|(s_id, p_id, o_id)| {
            let subject = self
                .dictionary
                .lookup_subject(s_id)
                .ok_or(HdtError::DictionaryDecodeError { id: s_id as u64 })?
                .to_owned();
            let predicate = self
                .dictionary
                .lookup_predicate(p_id)
                .ok_or(HdtError::DictionaryDecodeError { id: p_id as u64 })?
                .to_owned();
            let object = self
                .dictionary
                .lookup_object(o_id)
                .ok_or(HdtError::DictionaryDecodeError { id: o_id as u64 })?
                .to_owned();
            Ok(HdtTriple {
                subject,
                predicate,
                object,
            })
        })
    }

    /// Iterate all triples with string resolution — alias to `triples()`.
    pub fn iter_triples(&self) -> impl Iterator<Item = Result<(String, String, String), HdtError>> + '_ {
        self.triples().map(|r| r.map(|t| (t.subject, t.predicate, t.object)))
    }

    /// Resolve a subject ID to its string via the dictionary.
    ///
    /// # Errors
    /// Returns `HdtError::DictionaryDecodeError` if the ID is out of range.
    pub fn lookup_subject(&self, id: u32) -> Result<&str, HdtError> {
        self.dictionary
            .lookup_subject(id)
            .ok_or(HdtError::DictionaryDecodeError { id: id as u64 })
    }

    /// Resolve a predicate ID to its string via the dictionary.
    ///
    /// # Errors
    /// Returns `HdtError::DictionaryDecodeError` if the ID is out of range.
    pub fn lookup_predicate(&self, id: u32) -> Result<&str, HdtError> {
        self.dictionary
            .lookup_predicate(id)
            .ok_or(HdtError::DictionaryDecodeError { id: id as u64 })
    }

    /// Resolve an object ID to its string via the dictionary.
    ///
    /// # Errors
    /// Returns `HdtError::DictionaryDecodeError` if the ID is out of range.
    pub fn lookup_object(&self, id: u32) -> Result<&str, HdtError> {
        self.dictionary
            .lookup_object(id)
            .ok_or(HdtError::DictionaryDecodeError { id: id as u64 })
    }

    /// Find all triples where the subject matches `subject` string.
    ///
    /// Performs a reverse dictionary lookup to find the subject ID, then
    /// collects all triples with that ID from the triples section.
    pub fn lookup_subject_str(&self, subject: &str) -> Result<Vec<(String, String, String)>, HdtError> {
        // Find subject ID via dictionary
        let s_id = self.dictionary.subject_to_id(subject);
        match s_id {
            None => Ok(Vec::new()),
            Some(id) => {
                let result = self
                    .triples_section
                    .iter_ids()
                    .filter(|(s, _, _)| *s == id)
                    .map(|(s, p, o)| {
                        let subj = self
                            .dictionary
                            .lookup_subject(s)
                            .ok_or(HdtError::DictionaryDecodeError { id: s as u64 })?
                            .to_owned();
                        let pred = self
                            .dictionary
                            .lookup_predicate(p)
                            .ok_or(HdtError::DictionaryDecodeError { id: p as u64 })?
                            .to_owned();
                        let obj = self
                            .dictionary
                            .lookup_object(o)
                            .ok_or(HdtError::DictionaryDecodeError { id: o as u64 })?
                            .to_owned();
                        Ok((subj, pred, obj))
                    })
                    .collect::<Result<Vec<_>, HdtError>>()?;
                Ok(result)
            }
        }
    }

    // -----------------------------------------------------------------------
    // Internal parsing
    // -----------------------------------------------------------------------

    /// Master parser: validate magic, then parse header, dictionary, triples.
    fn parse_bytes(data: &[u8]) -> Result<Self, HdtError> {
        if data.len() < HDT_MAGIC_SIMPLE.len() || &data[..HDT_MAGIC_SIMPLE.len()] != HDT_MAGIC_SIMPLE {
            return Err(HdtError::InvalidMagic {
                got: data[..HDT_MAGIC_SIMPLE.len().min(data.len())].to_vec(),
            });
        }
        let mut offset = HDT_MAGIC_SIMPLE.len();

        let header = Self::parse_header(data, &mut offset)?;
        let dictionary = Self::parse_dictionary(data, &mut offset)?;
        let triples_section = Self::parse_triples(data, &mut offset)?;

        Ok(HdtReader {
            header,
            dictionary,
            triples_section,
        })
    }

    /// Parse the HDT header section.
    fn parse_header(data: &[u8], offset: &mut usize) -> Result<HdtHeader, HdtError> {
        if *offset + 8 > data.len() {
            return Err(HdtError::InvalidSection {
                name: "truncated before header size field".to_owned(),
            });
        }
        let hdr_size = u64::from_le_bytes(
            data[*offset..*offset + 8]
                .try_into()
                .map_err(|_| HdtError::InvalidSection {
                    name: "cannot read header size".to_owned(),
                })?,
        ) as usize;
        *offset += 8;

        let hdr_end = (*offset + hdr_size).min(data.len());
        let hdr_bytes = &data[*offset..hdr_end];
        *offset = hdr_end;

        let mut header = HdtHeader::default();
        let text = std::str::from_utf8(hdr_bytes).map_err(|e| HdtError::InvalidSection {
            name: format!("header UTF-8: {}", e),
        })?;

        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some((key, value)) = line.split_once('=') {
                match key.trim() {
                    "triples" => {
                        if let Ok(v) = value.trim().parse::<u64>() {
                            header.triples_count = v;
                        }
                    }
                    "subjects" => {
                        if let Ok(v) = value.trim().parse::<u64>() {
                            header.subjects_count = v;
                        }
                    }
                    "predicates" => {
                        if let Ok(v) = value.trim().parse::<u64>() {
                            header.predicates_count = v;
                        }
                    }
                    "objects" => {
                        if let Ok(v) = value.trim().parse::<u64>() {
                            header.objects_count = v;
                        }
                    }
                    "shared" => {
                        if let Ok(v) = value.trim().parse::<u64>() {
                            header.shared_count = v;
                        }
                    }
                    "format" => {
                        header.format = value.trim().to_owned();
                    }
                    "baseURI" | "base_uri" => {
                        header.base_uri = Some(value.trim().to_owned());
                    }
                    _ => {}
                }
            }
        }
        Ok(header)
    }

    /// Parse all four dictionary sections from `data` at `offset`.
    fn parse_dictionary(data: &[u8], offset: &mut usize) -> Result<HdtDictionary, HdtError> {
        let mut dict = HdtDictionary::new();

        dict.shared = Self::read_dict_section(data, offset, "shared")?;
        dict.subjects = Self::read_dict_section(data, offset, "subjects")?;
        dict.predicates = Self::read_dict_section(data, offset, "predicates")?;
        dict.objects = Self::read_dict_section(data, offset, "objects")?;

        Ok(dict)
    }

    /// Read one dictionary section: 4-byte LE length prefix + null-separated strings.
    fn read_dict_section(
        data: &[u8],
        offset: &mut usize,
        name: &str,
    ) -> Result<Vec<String>, HdtError> {
        if *offset + 4 > data.len() {
            return Err(HdtError::InvalidSection {
                name: format!("{} section: truncated before size", name),
            });
        }
        let sec_len = u32::from_le_bytes(
            data[*offset..*offset + 4]
                .try_into()
                .map_err(|_| HdtError::InvalidSection {
                    name: format!("{} section: cannot read 4-byte length", name),
                })?,
        ) as usize;
        *offset += 4;

        if *offset + sec_len > data.len() {
            return Err(HdtError::InvalidSection {
                name: format!(
                    "{} section: truncated (need {} bytes, have {})",
                    name,
                    sec_len,
                    data.len() - *offset
                ),
            });
        }

        let sec_data = &data[*offset..*offset + sec_len];
        *offset += sec_len;

        parse_plain_dictionary(sec_data)
    }

    /// Parse the triples section: delegates to `HdtTriplesSection::parse`.
    fn parse_triples(data: &[u8], offset: &mut usize) -> Result<HdtTriplesSection, HdtError> {
        let triples_data = &data[*offset..];
        *offset = data.len();
        HdtTriplesSection::parse(triples_data)
    }
}
