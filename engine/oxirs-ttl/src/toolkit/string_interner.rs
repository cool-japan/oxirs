//! String interning for efficient deduplication of repeated strings (IRIs, prefixes, language tags)
//!
//! This module provides a high-performance string interner specifically optimized for RDF parsing,
//! where many strings (especially IRIs, predicates, and prefixes) are repeated frequently.

use std::borrow::Cow;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// A string interner that deduplicates strings to reduce memory usage
///
/// This is particularly useful for RDF parsing where:
/// - Common predicates (rdf:type, rdfs:label, etc.) appear many times
/// - Namespace URIs are repeated frequently
/// - Language tags and datatypes are reused
///
/// # Example
///
/// ```
/// use oxirs_ttl::toolkit::StringInterner;
/// use std::sync::Arc;
///
/// let mut interner = StringInterner::new();
///
/// // These will all point to the same underlying string
/// let s1 = interner.intern("http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
/// let s2 = interner.intern("http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
///
/// // The Arc pointers should be the same (cheap to clone)
/// assert!(Arc::ptr_eq(&s1, &s2));
/// ```
#[derive(Debug, Clone)]
pub struct StringInterner {
    /// Map from string content to interned `Arc<String>`
    map: HashMap<InternedString, Arc<String>>,
    /// Statistics for monitoring performance
    stats: InternerStats,
}

/// Statistics about string interning performance
#[derive(Debug, Clone, Default)]
pub struct InternerStats {
    /// Total number of intern() calls
    pub total_requests: usize,
    /// Number of times a string was found in the cache (hit)
    pub cache_hits: usize,
    /// Number of times a new string was allocated (miss)
    pub cache_misses: usize,
    /// Total number of unique strings stored
    pub unique_strings: usize,
    /// Total bytes saved by deduplication (approximate)
    pub bytes_saved: usize,
}

/// Wrapper for hash map key that allows lookup by &str without allocating
#[derive(Debug, Clone, Eq)]
struct InternedString(Arc<String>);

impl InternedString {
    fn new(s: Arc<String>) -> Self {
        Self(s)
    }

    fn as_str(&self) -> &str {
        &self.0
    }
}

impl PartialEq for InternedString {
    fn eq(&self, other: &Self) -> bool {
        self.0.as_str() == other.0.as_str()
    }
}

impl Hash for InternedString {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.as_str().hash(state);
    }
}

impl std::borrow::Borrow<str> for InternedString {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl StringInterner {
    /// Create a new string interner
    pub fn new() -> Self {
        Self::with_capacity(1024)
    }

    /// Create a new string interner with pre-allocated capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            map: HashMap::with_capacity(capacity),
            stats: InternerStats::default(),
        }
    }

    /// Create a new interner pre-populated with common RDF namespaces
    pub fn with_common_namespaces() -> Self {
        let mut interner = Self::with_capacity(2048);

        // Pre-populate with common RDF namespaces
        let common_namespaces = [
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "http://www.w3.org/2000/01/rdf-schema#",
            "http://www.w3.org/2001/XMLSchema#",
            "http://www.w3.org/2002/07/owl#",
            "http://xmlns.com/foaf/0.1/",
            "http://purl.org/dc/elements/1.1/",
            "http://purl.org/dc/terms/",
            "http://schema.org/",
            "http://www.w3.org/ns/shacl#",
            "http://www.w3.org/2004/02/skos/core#",
            // Common predicates
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            "http://www.w3.org/2000/01/rdf-schema#label",
            "http://www.w3.org/2000/01/rdf-schema#comment",
            "http://www.w3.org/2000/01/rdf-schema#subClassOf",
            "http://www.w3.org/2000/01/rdf-schema#subPropertyOf",
            "http://www.w3.org/2000/01/rdf-schema#domain",
            "http://www.w3.org/2000/01/rdf-schema#range",
            "http://www.w3.org/2002/07/owl#sameAs",
            "http://www.w3.org/2002/07/owl#equivalentClass",
            "http://www.w3.org/2002/07/owl#equivalentProperty",
        ];

        for ns in &common_namespaces {
            interner.intern(ns);
        }

        // Reset stats after pre-population
        interner.stats = InternerStats {
            unique_strings: interner.map.len(),
            ..Default::default()
        };

        interner
    }

    /// Intern a string, returning an Arc to the deduplicated version
    ///
    /// If the string has been seen before, returns the existing Arc.
    /// Otherwise, allocates a new String and stores it.
    pub fn intern(&mut self, s: &str) -> Arc<String> {
        self.stats.total_requests += 1;

        // Try to find existing interned string
        if let Some(interned_key) = self.map.get_key_value(s) {
            self.stats.cache_hits += 1;
            // Estimate bytes saved: we would have allocated this string, but didn't
            self.stats.bytes_saved += s.len();
            return interned_key.0 .0.clone();
        }

        // Not found - allocate new string
        self.stats.cache_misses += 1;
        self.stats.unique_strings += 1;

        let arc_string = Arc::new(s.to_string());
        let key = InternedString::new(arc_string.clone());
        self.map.insert(key, arc_string.clone());

        arc_string
    }

    /// Intern a string only if it's likely to be repeated
    ///
    /// Uses heuristics to decide whether to intern:
    /// - Always intern if it contains common namespace patterns
    /// - Always intern if it's a common language tag
    /// - Otherwise, intern if length > threshold
    pub fn intern_if_beneficial<'a>(&mut self, s: &'a str, min_length: usize) -> Cow<'a, str> {
        // Always intern common patterns
        if s.contains("www.w3.org")
            || s.contains("schema.org")
            || s.contains("xmlns.com")
            || s.contains("purl.org")
            || matches!(s, "en" | "de" | "fr" | "es" | "ja" | "zh" | "ar" | "hi")
        {
            return Cow::Owned((*self.intern(s)).clone());
        }

        // Intern longer strings that are likely to be repeated
        if s.len() >= min_length {
            Cow::Owned((*self.intern(s)).clone())
        } else {
            Cow::Borrowed(s)
        }
    }

    /// Get the number of unique strings stored
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Check if the interner is empty
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Get interning statistics
    pub fn stats(&self) -> &InternerStats {
        &self.stats
    }

    /// Get the cache hit rate (0.0 to 1.0)
    pub fn hit_rate(&self) -> f64 {
        if self.stats.total_requests == 0 {
            return 0.0;
        }
        self.stats.cache_hits as f64 / self.stats.total_requests as f64
    }

    /// Clear all interned strings and reset statistics
    pub fn clear(&mut self) {
        self.map.clear();
        self.stats = InternerStats::default();
    }

    /// Estimate memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        // Each HashMap entry has overhead + key (Arc<String>) + value (Arc<String>)
        const HASHMAP_ENTRY_OVERHEAD: usize = 24; // Approximate
        const ARC_OVERHEAD: usize = 16; // Arc has refcount + pointer

        let mut total = 0;

        // HashMap structure overhead
        total += self.map.capacity() * HASHMAP_ENTRY_OVERHEAD;

        // String data + Arc overhead
        for key in self.map.keys() {
            total += ARC_OVERHEAD; // InternedString wrapper
            total += ARC_OVERHEAD; // Key Arc<String>
            total += ARC_OVERHEAD; // Value Arc<String>
            total += key.as_str().len(); // String data (counted once, shared by key and value)
            total += std::mem::size_of::<String>(); // String struct overhead
        }

        total
    }

    /// Shrink the hash map to fit the current number of entries
    pub fn shrink_to_fit(&mut self) {
        self.map.shrink_to_fit();
    }
}

impl Default for StringInterner {
    fn default() -> Self {
        Self::new()
    }
}

impl InternerStats {
    /// Get a human-readable report of interning statistics
    pub fn report(&self) -> String {
        let hit_rate = if self.total_requests > 0 {
            (self.cache_hits as f64 / self.total_requests as f64) * 100.0
        } else {
            0.0
        };

        format!(
            "String Interning Statistics:\n\
             - Total requests: {}\n\
             - Cache hits: {} ({:.1}%)\n\
             - Cache misses: {}\n\
             - Unique strings: {}\n\
             - Bytes saved: {} ({:.1} KB)",
            self.total_requests,
            self.cache_hits,
            hit_rate,
            self.cache_misses,
            self.unique_strings,
            self.bytes_saved,
            self.bytes_saved as f64 / 1024.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_interning() {
        let mut interner = StringInterner::new();

        let s1 = interner.intern("http://example.org/");
        let s2 = interner.intern("http://example.org/");

        // Should be the exact same Arc (pointer equality)
        assert!(Arc::ptr_eq(&s1, &s2));
        assert_eq!(interner.len(), 1);
    }

    #[test]
    fn test_different_strings() {
        let mut interner = StringInterner::new();

        let s1 = interner.intern("http://example.org/a");
        let s2 = interner.intern("http://example.org/b");

        // Should be different Arcs
        assert!(!Arc::ptr_eq(&s1, &s2));
        assert_eq!(interner.len(), 2);
    }

    #[test]
    fn test_stats() {
        let mut interner = StringInterner::new();

        interner.intern("test");
        interner.intern("test");
        interner.intern("test");
        interner.intern("other");

        let stats = interner.stats();
        assert_eq!(stats.total_requests, 4);
        assert_eq!(stats.cache_hits, 2); // "test" hit twice
        assert_eq!(stats.cache_misses, 2); // "test" and "other" each missed once
        assert_eq!(stats.unique_strings, 2);
    }

    #[test]
    fn test_hit_rate() {
        let mut interner = StringInterner::new();

        interner.intern("a");
        interner.intern("a");
        interner.intern("b");
        interner.intern("b");

        // 4 total requests, 2 hits (50%)
        assert_eq!(interner.hit_rate(), 0.5);
    }

    #[test]
    fn test_common_namespaces() {
        let interner = StringInterner::with_common_namespaces();

        // Should have pre-populated common namespaces
        assert!(interner.len() > 10);

        // Stats should be reset after pre-population
        assert_eq!(interner.stats().total_requests, 0);
        assert_eq!(interner.stats().cache_hits, 0);
    }

    #[test]
    fn test_intern_if_beneficial() {
        let mut interner = StringInterner::new();

        // Short string - should not intern
        let result = interner.intern_if_beneficial("ab", 10);
        assert!(matches!(result, Cow::Borrowed(_)));

        // Long string - should intern
        let result = interner.intern_if_beneficial("http://example.org/very/long/uri", 10);
        assert!(matches!(result, Cow::Owned(_)));

        // Common namespace - should always intern regardless of length
        let result = interner.intern_if_beneficial("www.w3.org", 100);
        assert!(matches!(result, Cow::Owned(_)));
    }

    #[test]
    fn test_memory_usage() {
        let mut interner = StringInterner::new();

        interner.intern("short");
        interner.intern("a much longer string that takes more memory");

        let usage = interner.memory_usage();
        assert!(usage > 0);
        println!("Memory usage: {} bytes", usage);
    }

    #[test]
    fn test_clear() {
        let mut interner = StringInterner::new();

        interner.intern("test1");
        interner.intern("test2");
        assert_eq!(interner.len(), 2);

        interner.clear();
        assert_eq!(interner.len(), 0);
        assert_eq!(interner.stats().total_requests, 0);
    }

    #[test]
    fn test_rdf_use_case() {
        let mut interner = StringInterner::with_common_namespaces();

        // Simulate parsing RDF with repeated predicates
        let rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";

        let initial_unique = interner.len();

        // Intern the same predicate many times (simulates parsing many triples)
        for _ in 0..1000 {
            interner.intern(rdf_type);
        }

        // Should still have the same number of unique strings
        assert_eq!(interner.len(), initial_unique);

        // Should have very high hit rate
        assert!(interner.hit_rate() > 0.95);

        println!("{}", interner.stats().report());
    }

    #[test]
    fn test_stats_report() {
        let mut interner = StringInterner::new();

        interner.intern("test");
        interner.intern("test");
        interner.intern("other");

        let report = interner.stats().report();
        assert!(report.contains("Total requests: 3"));
        assert!(report.contains("Cache hits: 1"));
        assert!(report.contains("Unique strings: 2"));
    }
}
