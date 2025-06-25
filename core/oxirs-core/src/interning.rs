//! String interning system for performance optimization
//!
//! This module provides efficient string interning for commonly used strings
//! like IRIs, datatype URIs, and other RDF terms. String interning reduces
//! memory usage and improves comparison performance by ensuring that equal
//! strings are stored only once and can be compared by pointer equality.

use std::collections::HashMap;
use std::sync::{Arc, RwLock, Weak};
use std::hash::{Hash, Hasher};

/// A thread-safe string interner that deduplicates strings
#[derive(Debug)]
pub struct StringInterner {
    /// Map from string content to weak references of interned strings
    strings: RwLock<HashMap<String, Weak<str>>>,
    /// Statistics for monitoring performance
    stats: RwLock<InternerStats>,
}

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
    /// Create a new string interner
    pub fn new() -> Self {
        StringInterner {
            strings: RwLock::new(HashMap::new()),
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
        self.strings.read().unwrap().len()
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

/// Global interner instances for common string types
lazy_static::lazy_static! {
    /// Global interner for IRI strings
    pub static ref IRI_INTERNER: StringInterner = StringInterner::new();
    
    /// Global interner for datatype IRIs
    pub static ref DATATYPE_INTERNER: StringInterner = StringInterner::new();
    
    /// Global interner for language tags
    pub static ref LANGUAGE_INTERNER: StringInterner = StringInterner::new();
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

        assert_eq!(string_type.as_str(), "http://www.w3.org/2001/XMLSchema#string");
        assert_eq!(integer_type.as_str(), "http://www.w3.org/2001/XMLSchema#integer");
        assert_eq!(rdf_type.as_str(), "http://www.w3.org/1999/02/22-rdf-syntax-ns#type");

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
}