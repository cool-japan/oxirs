//! # HDT Dictionary
//!
//! Four-section string dictionary mapping integer IDs to RDF term strings.
//!
//! The HDT plain dictionary format stores null-separated UTF-8 string lists in
//! four sections:
//! - **Shared** (SH): IRIs that appear as both subject and object.
//! - **Subjects** (SO): IRIs that appear only as subjects.
//! - **Predicates** (P): All predicate IRIs.
//! - **Objects** (O): IRIs / literals that appear only as objects.
//!
//! ## ID Addressing
//!
//! | Term role | ID range                                        | Resolved from     |
//! |-----------|--------------------------------------------------|-------------------|
//! | Subject   | 1 … shared.len()                                 | `shared[id - 1]`  |
//! | Subject   | shared.len()+1 … shared.len()+subjects.len()    | `subjects[id - shared.len() - 1]` |
//! | Predicate | 1 … predicates.len()                            | `predicates[id - 1]` |
//! | Object    | 1 … shared.len()                                 | `shared[id - 1]`  |
//! | Object    | shared.len()+1 … shared.len()+objects.len()     | `objects[id - shared.len() - 1]` |

use super::HdtError;
use super::format::read_vbyte_slice;

// ---------------------------------------------------------------------------
// DictionarySection
// ---------------------------------------------------------------------------

/// A single decoded section of an HDT dictionary.
///
/// HDT dictionaries use either:
/// - **Plain** encoding: null-terminated UTF-8 strings in sorted order.
/// - **Front-coded** (PSFC) encoding: every k-th entry is stored fully;
///   subsequent entries store only the suffix after a shared prefix, encoded
///   as `(shared_prefix_len_vbyte)(suffix_bytes\0)`.
#[derive(Debug, Clone, Default)]
pub struct DictionarySection {
    /// All decoded terms in sorted order.  IDs are 1-based indexes into this
    /// vector (i.e. `terms[0]` has ID 1).
    pub terms: Vec<String>,
}

impl DictionarySection {
    /// Create an empty section.
    pub fn new() -> Self {
        Self::default()
    }

    /// Decode a plain (null-separated) dictionary section.
    ///
    /// # Errors
    /// Returns `HdtError::DictionaryDecodeError` (with `id = 0`) if a
    /// segment contains invalid UTF-8.
    pub fn from_plain(data: &[u8]) -> Result<Self, HdtError> {
        if data.is_empty() {
            return Ok(Self::default());
        }
        let mut terms = Vec::new();
        for segment in data.split(|b| *b == 0) {
            if segment.is_empty() {
                continue;
            }
            let s = std::str::from_utf8(segment).map_err(|_| HdtError::DictionaryDecodeError { id: 0 })?;
            terms.push(s.to_owned());
        }
        Ok(DictionarySection { terms })
    }

    /// Decode a front-coded (PSFC) dictionary section.
    ///
    /// In PSFC encoding every `k`-th entry (0-indexed: entries 0, k, 2k, …) is
    /// stored as a complete null-terminated string.  Each non-anchor entry is
    /// stored as:
    /// ```text
    /// [vbyte shared_prefix_length][unique_suffix_bytes\0]
    /// ```
    /// where `shared_prefix_length` is the number of bytes from the previous
    /// entry that are re-used.
    ///
    /// # Errors
    /// Returns `HdtError::DictionaryDecodeError` on truncated/invalid data.
    pub fn from_front_coded(data: &[u8], k: usize) -> Result<Self, HdtError> {
        if data.is_empty() {
            return Ok(Self::default());
        }
        let k = k.max(1);
        let mut terms: Vec<String> = Vec::new();
        let mut pos = 0usize;

        while pos < data.len() {
            let entry_idx = terms.len();

            if entry_idx % k == 0 {
                // Anchor entry: full null-terminated string
                let null_pos = data[pos..]
                    .iter()
                    .position(|b| *b == 0)
                    .ok_or(HdtError::DictionaryDecodeError { id: entry_idx as u64 })?;
                let s = std::str::from_utf8(&data[pos..pos + null_pos])
                    .map_err(|_| HdtError::DictionaryDecodeError { id: entry_idx as u64 })?;
                terms.push(s.to_owned());
                pos += null_pos + 1; // skip the null
            } else {
                // Delta entry: vbyte prefix length + suffix\0
                let (shared_len, consumed) = read_vbyte_slice(&data[pos..])?;
                pos += consumed;
                let shared_len = shared_len as usize;

                let null_pos = data[pos..]
                    .iter()
                    .position(|b| *b == 0)
                    .ok_or(HdtError::DictionaryDecodeError { id: entry_idx as u64 })?;
                let suffix = std::str::from_utf8(&data[pos..pos + null_pos])
                    .map_err(|_| HdtError::DictionaryDecodeError { id: entry_idx as u64 })?;
                pos += null_pos + 1;

                // The previous entry gives us the shared prefix
                let prev = terms
                    .last()
                    .ok_or(HdtError::DictionaryDecodeError { id: entry_idx as u64 })?;
                let prefix_bytes = prev.as_bytes().get(..shared_len).ok_or(
                    HdtError::DictionaryDecodeError { id: entry_idx as u64 },
                )?;
                let prefix = std::str::from_utf8(prefix_bytes)
                    .map_err(|_| HdtError::DictionaryDecodeError { id: entry_idx as u64 })?;

                terms.push(format!("{}{}", prefix, suffix));
            }
        }

        Ok(DictionarySection { terms })
    }

    /// Return the term for a 1-based ID, or `None` if out of range.
    pub fn id_to_term(&self, id: usize) -> Option<&str> {
        if id == 0 {
            return None;
        }
        self.terms.get(id - 1).map(String::as_str)
    }

    /// Find the 1-based ID for `term` using binary search.
    ///
    /// Returns `None` if the term is not present.
    pub fn term_to_id(&self, term: &str) -> Option<usize> {
        self.terms
            .binary_search_by(|s| s.as_str().cmp(term))
            .ok()
            .map(|idx| idx + 1)
    }
}

// ---------------------------------------------------------------------------
// HdtDictionary
// ---------------------------------------------------------------------------

/// In-memory HDT dictionary with four independent string sections.
#[derive(Debug, Clone, Default)]
pub struct HdtDictionary {
    /// Strings shared between subject and object positions.
    pub shared: Vec<String>,
    /// Strings used only in subject position (after shared IDs).
    pub subjects: Vec<String>,
    /// All predicate strings (1-indexed).
    pub predicates: Vec<String>,
    /// Strings used only in object position (after shared IDs).
    pub objects: Vec<String>,
}

impl HdtDictionary {
    /// Create an empty dictionary.
    pub fn new() -> Self {
        Self::default()
    }

    // -----------------------------------------------------------------------
    // ID → term lookups
    // -----------------------------------------------------------------------

    /// Resolve a subject ID to the corresponding string.
    ///
    /// - IDs `1..=shared.len()` map to `shared`.
    /// - IDs `shared.len()+1..=shared.len()+subjects.len()` map to `subjects`.
    /// - Returns `None` for ID 0 or out-of-range IDs.
    pub fn lookup_subject(&self, id: u32) -> Option<&str> {
        if id == 0 {
            return None;
        }
        let id_usize = id as usize;
        let sh_len = self.shared.len();
        if id_usize <= sh_len {
            return self.shared.get(id_usize - 1).map(String::as_str);
        }
        let so_idx = id_usize - sh_len - 1;
        self.subjects.get(so_idx).map(String::as_str)
    }

    /// Resolve a predicate ID to the corresponding string.
    ///
    /// IDs are 1-indexed directly into `predicates`.
    /// Returns `None` for ID 0 or out-of-range IDs.
    pub fn lookup_predicate(&self, id: u32) -> Option<&str> {
        if id == 0 {
            return None;
        }
        self.predicates.get(id as usize - 1).map(String::as_str)
    }

    /// Resolve an object ID to the corresponding string.
    ///
    /// - IDs `1..=shared.len()` map to `shared`.
    /// - IDs `shared.len()+1..=shared.len()+objects.len()` map to `objects`.
    /// - Returns `None` for ID 0 or out-of-range IDs.
    pub fn lookup_object(&self, id: u32) -> Option<&str> {
        if id == 0 {
            return None;
        }
        let id_usize = id as usize;
        let sh_len = self.shared.len();
        if id_usize <= sh_len {
            return self.shared.get(id_usize - 1).map(String::as_str);
        }
        let o_idx = id_usize - sh_len - 1;
        self.objects.get(o_idx).map(String::as_str)
    }

    // -----------------------------------------------------------------------
    // term → ID reverse lookups
    // -----------------------------------------------------------------------

    /// Find the subject ID for `s` (binary search in shared then subjects).
    ///
    /// Returns `None` if the term is not found.
    pub fn subject_to_id(&self, s: &str) -> Option<u32> {
        // Check shared first (IDs 1..=shared.len())
        if let Ok(idx) = self.shared.binary_search_by(|t| t.as_str().cmp(s)) {
            return Some((idx + 1) as u32);
        }
        // Then subjects-only (IDs shared.len()+1..=shared.len()+subjects.len())
        if let Ok(idx) = self.subjects.binary_search_by(|t| t.as_str().cmp(s)) {
            return Some((self.shared.len() + idx + 1) as u32);
        }
        None
    }

    /// Find the predicate ID for `p` (binary search in predicates).
    pub fn predicate_to_id(&self, p: &str) -> Option<u32> {
        self.predicates
            .binary_search_by(|t| t.as_str().cmp(p))
            .ok()
            .map(|idx| (idx + 1) as u32)
    }

    /// Find the object ID for `o` (binary search in shared then objects).
    pub fn object_to_id(&self, o: &str) -> Option<u32> {
        if let Ok(idx) = self.shared.binary_search_by(|t| t.as_str().cmp(o)) {
            return Some((idx + 1) as u32);
        }
        if let Ok(idx) = self.objects.binary_search_by(|t| t.as_str().cmp(o)) {
            return Some((self.shared.len() + idx + 1) as u32);
        }
        None
    }

    // -----------------------------------------------------------------------
    // Count helpers
    // -----------------------------------------------------------------------

    /// Total number of IDs addressable as a subject (shared + subjects-only).
    pub fn subject_count(&self) -> u32 {
        (self.shared.len() + self.subjects.len()) as u32
    }

    /// Total number of predicate IDs.
    pub fn predicate_count(&self) -> u32 {
        self.predicates.len() as u32
    }

    /// Total number of IDs addressable as an object (shared + objects-only).
    pub fn object_count(&self) -> u32 {
        (self.shared.len() + self.objects.len()) as u32
    }
}

// ---------------------------------------------------------------------------
// parse_plain_dictionary  (kept for backward compatibility)
// ---------------------------------------------------------------------------

/// Parse the "plain" HDT dictionary section encoding: a null-separated (`\0`)
/// list of UTF-8 strings.
///
/// An empty `data` slice produces an empty vector.
///
/// # Errors
/// Returns `HdtError::DictionaryDecodeError` (id=0) if a segment contains
/// invalid UTF-8.
pub fn parse_plain_dictionary(data: &[u8]) -> Result<Vec<String>, HdtError> {
    DictionarySection::from_plain(data).map(|s| s.terms)
}
