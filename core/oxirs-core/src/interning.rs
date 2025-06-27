//! String interning system for performance optimization
//!
//! This module provides efficient string interning for commonly used strings
//! like IRIs, datatype URIs, and other RDF terms. String interning reduces
//! memory usage and improves comparison performance by ensuring that equal
//! strings are stored only once and can be compared by pointer equality.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock, Weak};

/// A thread-safe string interner that deduplicates strings with ID mapping
#[derive(Debug)]
pub struct StringInterner {
    /// Map from string content to weak references of interned strings
    strings: RwLock<HashMap<String, Weak<str>>>,
    /// Bidirectional mapping between strings and numeric IDs
    string_to_id: RwLock<HashMap<String, u32>>,
    /// Map from ID back to string
    id_to_string: RwLock<HashMap<u32, Arc<str>>>,
    /// Next available ID
    next_id: AtomicU32,
    /// Statistics for monitoring performance
    stats: RwLock<InternerStats>,
}

/// Atomic 32-bit unsigned integer type for thread-safe ID generation
use std::sync::atomic::AtomicU32;

/// Statistics for monitoring string interner performance
#[derive(Debug, Clone, Default)]
pub struct InternerStats {
    pub total_requests: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub total_strings_stored: usize,
    pub memory_saved_bytes: usize,
}

impl InternerStats {
    pub fn hit_ratio(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.cache_hits as f64 / self.total_requests as f64
        }
    }
}

impl StringInterner {
    /// Create a new string interner with ID mapping
    pub fn new() -> Self {
        StringInterner {
            strings: RwLock::new(HashMap::new()),
            string_to_id: RwLock::new(HashMap::new()),
            id_to_string: RwLock::new(HashMap::new()),
            next_id: AtomicU32::new(0),
            stats: RwLock::new(InternerStats::default()),
        }
    }

    /// Intern a string, returning an Arc<str> that can be cheaply cloned and compared
    pub fn intern(&self, s: &str) -> Arc<str> {
        // Fast path: try to get existing string with read lock
        {
            let strings = self.strings.read().unwrap();
            if let Some(weak_ref) = strings.get(s) {
                if let Some(arc_str) = weak_ref.upgrade() {
                    // Update stats
                    {
                        let mut stats = self.stats.write().unwrap();
                        stats.total_requests += 1;
                        stats.cache_hits += 1;
                    }
                    return arc_str;
                }
            }
        }

        // Slow path: need to create new string with write lock
        let mut strings = self.strings.write().unwrap();

        // Double-check in case another thread added it while we were waiting
        if let Some(weak_ref) = strings.get(s) {
            if let Some(arc_str) = weak_ref.upgrade() {
                // Update stats
                drop(strings); // Release write lock early
                {
                    let mut stats = self.stats.write().unwrap();
                    stats.total_requests += 1;
                    stats.cache_hits += 1;
                }
                return arc_str;
            }
        }

        // Create new interned string
        let arc_str: Arc<str> = Arc::from(s);
        let weak_ref = Arc::downgrade(&arc_str);
        strings.insert(s.to_string(), weak_ref);

        // Update stats
        drop(strings); // Release write lock early
        {
            let mut stats = self.stats.write().unwrap();
            stats.total_requests += 1;
            stats.cache_misses += 1;
            stats.total_strings_stored += 1;
            stats.memory_saved_bytes += s.len(); // Approximate memory saved on subsequent hits
        }

        arc_str
    }

    /// Intern a string and return both the Arc<str> and its numeric ID
    pub fn intern_with_id(&self, s: &str) -> (Arc<str>, u32) {
        // Fast path: check if we already have this string and its ID
        {
            let string_to_id = self.string_to_id.read().unwrap();
            if let Some(&id) = string_to_id.get(s) {
                // We have the ID, now get the Arc<str>
                let id_to_string = self.id_to_string.read().unwrap();
                if let Some(arc_str) = id_to_string.get(&id) {
                    // Update stats
                    {
                        let mut stats = self.stats.write().unwrap();
                        stats.total_requests += 1;
                        stats.cache_hits += 1;
                    }
                    return (arc_str.clone(), id);
                }
            }
        }

        // Slow path: need to create new entry
        let arc_str = self.intern(s); // This will handle the string interning
        let id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Update the ID mappings
        {
            let mut string_to_id = self.string_to_id.write().unwrap();
            string_to_id.insert(s.to_string(), id);
        }
        {
            let mut id_to_string = self.id_to_string.write().unwrap();
            id_to_string.insert(id, arc_str.clone());
        }

        (arc_str, id)
    }

    /// Get the ID for a string if it's already interned
    pub fn get_id(&self, s: &str) -> Option<u32> {
        let string_to_id = self.string_to_id.read().unwrap();
        string_to_id.get(s).copied()
    }

    /// Get the string for an ID if it exists
    pub fn get_string(&self, id: u32) -> Option<Arc<str>> {
        let id_to_string = self.id_to_string.read().unwrap();
        id_to_string.get(&id).cloned()
    }

    /// Get all ID mappings (useful for serialization/debugging)
    pub fn get_all_mappings(&self) -> Vec<(u32, Arc<str>)> {
        let id_to_string = self.id_to_string.read().unwrap();
        id_to_string
            .iter()
            .map(|(&id, s)| (id, s.clone()))
            .collect()
    }

    /// Clean up expired weak references to save memory
    pub fn cleanup(&self) {
        let mut strings = self.strings.write().unwrap();
        strings.retain(|_, weak_ref| weak_ref.strong_count() > 0);
    }

    /// Get current statistics
    pub fn stats(&self) -> InternerStats {
        self.stats.read().unwrap().clone()
    }

    /// Get the number of unique strings currently stored
    pub fn len(&self) -> usize {
        // Return the maximum of both counts to handle mixed usage
        let id_count = self.string_to_id.read().unwrap().len();
        let string_count = self.strings.read().unwrap().len();
        std::cmp::max(id_count, string_count)
    }

    /// Get the number of strings with ID mappings
    pub fn id_mapping_count(&self) -> usize {
        self.string_to_id.read().unwrap().len()
    }

    /// Check if the interner is empty
    pub fn is_empty(&self) -> bool {
        self.strings.read().unwrap().is_empty()
    }
}

impl Default for StringInterner {
    fn default() -> Self {
        Self::new()
    }
}

lazy_static::lazy_static! {
    // Global interner instances for common string types
    /// Global interner for IRI strings
    pub static ref IRI_INTERNER: StringInterner = StringInterner::new();

    /// Global interner for datatype IRIs
    pub static ref DATATYPE_INTERNER: StringInterner = StringInterner::new();

    /// Global interner for language tags
    pub static ref LANGUAGE_INTERNER: StringInterner = StringInterner::new();
    
    /// Global interner for general strings (JSON-LD processing)
    pub static ref STRING_INTERNER: StringInterner = StringInterner::new();
}

/// An interned string that supports efficient comparison and hashing
#[derive(Debug, Clone)]
pub struct InternedString {
    inner: Arc<str>,
}

impl InternedString {
    /// Create a new interned string using the default IRI interner
    pub fn new(s: &str) -> Self {
        InternedString {
            inner: IRI_INTERNER.intern(s),
        }
    }

    /// Create a new interned string using a specific interner
    pub fn new_with_interner(s: &str, interner: &StringInterner) -> Self {
        InternedString {
            inner: interner.intern(s),
        }
    }

    /// Create an interned datatype string
    pub fn new_datatype(s: &str) -> Self {
        InternedString {
            inner: DATATYPE_INTERNER.intern(s),
        }
    }

    /// Create an interned language tag string
    pub fn new_language(s: &str) -> Self {
        InternedString {
            inner: LANGUAGE_INTERNER.intern(s),
        }
    }

    /// Get the string content
    pub fn as_str(&self) -> &str {
        &self.inner
    }

    /// Get the inner Arc<str> for zero-copy operations
    pub fn as_arc_str(&self) -> &Arc<str> {
        &self.inner
    }

    /// Convert into the inner Arc<str>
    pub fn into_arc_str(self) -> Arc<str> {
        self.inner
    }
}

impl std::fmt::Display for InternedString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.inner)
    }
}

impl std::ops::Deref for InternedString {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl AsRef<str> for InternedString {
    fn as_ref(&self) -> &str {
        &self.inner
    }
}

impl PartialEq for InternedString {
    fn eq(&self, other: &Self) -> bool {
        // Fast pointer comparison first
        Arc::ptr_eq(&self.inner, &other.inner) || self.inner == other.inner
    }
}

impl Eq for InternedString {}

impl Hash for InternedString {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash the string content, not the pointer
        self.inner.hash(state);
    }
}

impl PartialOrd for InternedString {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for InternedString {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.inner.cmp(&other.inner)
    }
}

impl From<&str> for InternedString {
    fn from(s: &str) -> Self {
        InternedString::new(s)
    }
}

impl From<String> for InternedString {
    fn from(s: String) -> Self {
        InternedString::new(&s)
    }
}

/// Extension trait for string interning common RDF vocabulary
pub trait RdfVocabulary {
    /// Common XSD namespace
    const XSD_NS: &'static str = "http://www.w3.org/2001/XMLSchema#";
    /// Common RDF namespace
    const RDF_NS: &'static str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
    /// Common RDFS namespace
    const RDFS_NS: &'static str = "http://www.w3.org/2000/01/rdf-schema#";
    /// Common OWL namespace  
    const OWL_NS: &'static str = "http://www.w3.org/2002/07/owl#";

    fn xsd_string() -> InternedString {
        InternedString::new_datatype(&format!("{}string", Self::XSD_NS))
    }

    fn xsd_integer() -> InternedString {
        InternedString::new_datatype(&format!("{}integer", Self::XSD_NS))
    }

    fn xsd_decimal() -> InternedString {
        InternedString::new_datatype(&format!("{}decimal", Self::XSD_NS))
    }

    fn xsd_boolean() -> InternedString {
        InternedString::new_datatype(&format!("{}boolean", Self::XSD_NS))
    }

    fn xsd_double() -> InternedString {
        InternedString::new_datatype(&format!("{}double", Self::XSD_NS))
    }

    fn xsd_float() -> InternedString {
        InternedString::new_datatype(&format!("{}float", Self::XSD_NS))
    }

    fn xsd_date_time() -> InternedString {
        InternedString::new_datatype(&format!("{}dateTime", Self::XSD_NS))
    }

    fn rdf_type() -> InternedString {
        InternedString::new(&format!("{}type", Self::RDF_NS))
    }

    fn rdfs_label() -> InternedString {
        InternedString::new(&format!("{}label", Self::RDFS_NS))
    }

    fn rdfs_comment() -> InternedString {
        InternedString::new(&format!("{}comment", Self::RDFS_NS))
    }
}

/// Implement RdfVocabulary for InternedString to provide easy access to common terms
impl RdfVocabulary for InternedString {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_interner_basic() {
        let interner = StringInterner::new();

        let s1 = interner.intern("http://example.org/test");
        let s2 = interner.intern("http://example.org/test");
        let s3 = interner.intern("http://example.org/different");

        // Same string should return same Arc (pointer equality)
        assert!(Arc::ptr_eq(&s1, &s2));
        assert!(!Arc::ptr_eq(&s1, &s3));

        // Content should be equal
        assert_eq!(s1.as_ref(), "http://example.org/test");
        assert_eq!(s2.as_ref(), "http://example.org/test");
        assert_eq!(s3.as_ref(), "http://example.org/different");
    }

    #[test]
    fn test_string_interner_stats() {
        let interner = StringInterner::new();

        // First request - cache miss
        let _s1 = interner.intern("test");
        let stats = interner.stats();
        assert_eq!(stats.total_requests, 1);
        assert_eq!(stats.cache_misses, 1);
        assert_eq!(stats.cache_hits, 0);

        // Second request for same string - cache hit
        let _s2 = interner.intern("test");
        let stats = interner.stats();
        assert_eq!(stats.total_requests, 2);
        assert_eq!(stats.cache_misses, 1);
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.hit_ratio(), 0.5);
    }

    #[test]
    fn test_string_interner_cleanup() {
        let interner = StringInterner::new();

        {
            let _s1 = interner.intern("temporary");
            assert_eq!(interner.len(), 1);
        } // s1 goes out of scope

        interner.cleanup();
        assert_eq!(interner.len(), 0);
    }

    #[test]
    fn test_interned_string_creation() {
        let s1 = InternedString::new("http://example.org/test");
        let s2 = InternedString::new("http://example.org/test");
        let s3 = InternedString::new("http://example.org/different");

        assert_eq!(s1, s2);
        assert_ne!(s1, s3);
        assert_eq!(s1.as_str(), "http://example.org/test");
    }

    #[test]
    fn test_interned_string_ordering() {
        let s1 = InternedString::new("apple");
        let s2 = InternedString::new("banana");
        let s3 = InternedString::new("apple");

        assert!(s1 < s2);
        assert!(s2 > s1);
        assert_eq!(s1, s3);

        // Test that ordering is consistent
        let mut strings = vec![s2.clone(), s1.clone(), s3.clone()];
        strings.sort();
        assert_eq!(strings, vec![s1, s3, s2]);
    }

    #[test]
    fn test_interned_string_hashing() {
        use std::collections::HashMap;

        let s1 = InternedString::new("test");
        let s2 = InternedString::new("test");
        let s3 = InternedString::new("different");

        let mut map = HashMap::new();
        map.insert(s1.clone(), "value1");
        map.insert(s3.clone(), "value2");

        // s2 should map to the same value as s1 since they're equal
        assert_eq!(map.get(&s2), Some(&"value1"));
        assert_eq!(map.get(&s3), Some(&"value2"));
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn test_global_interners() {
        let iri1 = InternedString::new("http://example.org/test");
        let iri2 = InternedString::new("http://example.org/test");

        let datatype1 = InternedString::new_datatype("http://www.w3.org/2001/XMLSchema#string");
        let datatype2 = InternedString::new_datatype("http://www.w3.org/2001/XMLSchema#string");

        let lang1 = InternedString::new_language("en");
        let lang2 = InternedString::new_language("en");

        // Verify that equal strings are interned
        assert_eq!(iri1, iri2);
        assert_eq!(datatype1, datatype2);
        assert_eq!(lang1, lang2);
    }

    #[test]
    fn test_rdf_vocabulary() {
        let string_type = InternedString::xsd_string();
        let integer_type = InternedString::xsd_integer();
        let rdf_type = InternedString::rdf_type();

        assert_eq!(
            string_type.as_str(),
            "http://www.w3.org/2001/XMLSchema#string"
        );
        assert_eq!(
            integer_type.as_str(),
            "http://www.w3.org/2001/XMLSchema#integer"
        );
        assert_eq!(
            rdf_type.as_str(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
        );

        // Test that repeated calls return interned strings
        let string_type2 = InternedString::xsd_string();
        assert_eq!(string_type, string_type2);
    }

    #[test]
    fn test_interned_string_display() {
        let s = InternedString::new("http://example.org/test");
        assert_eq!(format!("{}", s), "http://example.org/test");
    }

    #[test]
    fn test_interned_string_deref() {
        let s = InternedString::new("test");
        assert_eq!(&*s, "test");
        assert_eq!(s.len(), 4);
        assert!(s.starts_with("te"));
    }

    #[test]
    fn test_interned_string_conversions() {
        let s1 = InternedString::from("test");
        let s2 = InternedString::from("test".to_string());

        assert_eq!(s1, s2);
        assert_eq!(s1.as_str(), "test");
    }

    #[test]
    fn test_concurrent_interning() {
        use std::sync::Arc;
        use std::thread;

        let interner = Arc::new(StringInterner::new());
        let handles: Vec<_> = (0..10)
            .map(|i| {
                let interner = Arc::clone(&interner);
                thread::spawn(move || {
                    let s = format!("http://example.org/test{}", i % 3);
                    (0..100).map(|_| interner.intern(&s)).collect::<Vec<_>>()
                })
            })
            .collect();

        let results: Vec<Vec<Arc<str>>> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // Verify that all equal strings are the same Arc
        for result_set in &results {
            for (i, s1) in result_set.iter().enumerate() {
                for s2 in &result_set[i + 1..] {
                    if s1.as_ref() == s2.as_ref() {
                        assert!(Arc::ptr_eq(s1, s2));
                    }
                }
            }
        }

        // Should have at most 3 unique strings (test0, test1, test2)
        assert!(interner.len() <= 3);
    }

    #[test]
    fn test_term_id_mapping() {
        let interner = StringInterner::new();

        // Test interning with ID
        let (arc1, id1) = interner.intern_with_id("test_string");
        let (arc2, id2) = interner.intern_with_id("test_string");

        // Same string should get same ID
        assert_eq!(id1, id2);
        assert!(Arc::ptr_eq(&arc1, &arc2));

        // Different strings should get different IDs
        let (arc3, id3) = interner.intern_with_id("different_string");
        assert_ne!(id1, id3);
        assert!(!Arc::ptr_eq(&arc1, &arc3));

        // Test ID lookup
        assert_eq!(interner.get_id("test_string"), Some(id1));
        assert_eq!(interner.get_id("different_string"), Some(id3));
        assert_eq!(interner.get_id("nonexistent"), None);

        // Test string lookup
        assert_eq!(interner.get_string(id1).unwrap().as_ref(), "test_string");
        assert_eq!(
            interner.get_string(id3).unwrap().as_ref(),
            "different_string"
        );
        assert_eq!(interner.get_string(999), None);
    }

    #[test]
    fn test_id_mapping_stats() {
        let interner = StringInterner::new();

        assert_eq!(interner.id_mapping_count(), 0);

        interner.intern_with_id("string1");
        assert_eq!(interner.id_mapping_count(), 1);

        interner.intern_with_id("string2");
        assert_eq!(interner.id_mapping_count(), 2);

        // Interning same string again shouldn't increase count
        interner.intern_with_id("string1");
        assert_eq!(interner.id_mapping_count(), 2);
    }

    #[test]
    fn test_get_all_mappings() {
        let interner = StringInterner::new();

        let (_, id1) = interner.intern_with_id("first");
        let (_, id2) = interner.intern_with_id("second");
        let (_, id3) = interner.intern_with_id("third");

        let mappings = interner.get_all_mappings();
        assert_eq!(mappings.len(), 3);

        // Verify all mappings are present
        let mut found_ids = vec![false; 3];
        for (id, string) in mappings {
            match string.as_ref() {
                "first" => {
                    assert_eq!(id, id1);
                    found_ids[0] = true;
                }
                "second" => {
                    assert_eq!(id, id2);
                    found_ids[1] = true;
                }
                "third" => {
                    assert_eq!(id, id3);
                    found_ids[2] = true;
                }
                _ => panic!("Unexpected string in mappings"),
            }
        }
        assert!(found_ids.iter().all(|&found| found));
    }

    #[test]
    fn test_mixed_interning_modes() {
        let interner = StringInterner::new();

        // Mix regular interning and ID interning
        let arc1 = interner.intern("regular");
        let (arc2, id2) = interner.intern_with_id("with_id");
        let arc3 = interner.intern("regular"); // Same as first

        // Regular interning should still work
        assert!(Arc::ptr_eq(&arc1, &arc3));

        // ID interning should work independently
        assert_eq!(interner.get_string(id2).unwrap().as_ref(), "with_id");

        // Mixed mode length reporting should work
        assert!(interner.len() >= 2);
    }
}
