//! # Term Dictionary with String Interning
//!
//! High-performance string dictionary implementation with automatic garbage collection,
//! reference counting, and memory management for efficient RDF term storage.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock, Weak};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Dictionary entry containing the interned string and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DictionaryEntry {
    /// The interned string value
    pub value: String,
    /// Reference count for garbage collection
    pub ref_count: usize,
    /// Creation timestamp
    pub created_at: u64,
    /// Last access timestamp
    pub last_accessed: u64,
    /// Size in bytes
    pub size: usize,
    /// Hash value for quick comparison
    pub hash: u64,
}

impl DictionaryEntry {
    /// Create a new dictionary entry
    pub fn new(value: String) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let hash = Self::calculate_hash(&value);
        let size = value.len();

        Self {
            value,
            ref_count: 1,
            created_at: now,
            last_accessed: now,
            size,
            hash,
        }
    }

    /// Calculate hash for a string
    fn calculate_hash(s: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        hasher.finish()
    }

    /// Increment reference count
    pub fn increment_ref(&mut self) {
        self.ref_count += 1;
        self.update_access_time();
    }

    /// Decrement reference count
    pub fn decrement_ref(&mut self) -> bool {
        if self.ref_count > 0 {
            self.ref_count -= 1;
        }
        self.ref_count == 0
    }

    /// Update last access time
    pub fn update_access_time(&mut self) {
        self.last_accessed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
    }

    /// Check if entry is eligible for garbage collection
    pub fn is_gc_eligible(&self, min_age_seconds: u64, max_idle_seconds: u64) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Must have zero references
        if self.ref_count > 0 {
            return false;
        }

        // Must be old enough
        let age = now.saturating_sub(self.created_at);
        if age < min_age_seconds {
            return false;
        }

        // Must be idle long enough
        let idle_time = now.saturating_sub(self.last_accessed);
        idle_time >= max_idle_seconds
    }
}

/// Dictionary configuration
#[derive(Debug, Clone)]
pub struct DictionaryConfig {
    /// Maximum number of entries before triggering cleanup
    pub max_entries: usize,
    /// Garbage collection interval in seconds
    pub gc_interval_seconds: u64,
    /// Minimum age for garbage collection eligibility (seconds)
    pub min_gc_age_seconds: u64,
    /// Maximum idle time before garbage collection (seconds)
    pub max_idle_seconds: u64,
    /// Enable automatic garbage collection
    pub enable_auto_gc: bool,
    /// Memory limit in bytes
    pub memory_limit_bytes: usize,
    /// Hash table load factor threshold
    pub load_factor_threshold: f64,
}

impl Default for DictionaryConfig {
    fn default() -> Self {
        Self {
            max_entries: 1_000_000,
            gc_interval_seconds: 60, // 1 minute
            min_gc_age_seconds: 300, // 5 minutes
            max_idle_seconds: 1800,  // 30 minutes
            enable_auto_gc: true,
            memory_limit_bytes: 1024 * 1024 * 1024, // 1GB
            load_factor_threshold: 0.75,
        }
    }
}

/// Dictionary statistics
#[derive(Debug, Clone, Default, Serialize)]
pub struct DictionaryStats {
    pub total_entries: usize,
    pub total_memory_bytes: usize,
    pub active_references: usize,
    pub gc_runs: u64,
    pub gc_entries_collected: u64,
    pub last_gc_time: u64,
    pub lookup_count: u64,
    pub hit_count: u64,
    pub miss_count: u64,
    pub collision_count: u64,
    pub avg_string_length: f64,
    pub max_string_length: usize,
    pub load_factor: f64,
}

/// String ID type for interned strings
pub type StringId = u64;

/// Interned string handle with automatic reference counting
pub struct InternedString {
    id: StringId,
    dictionary: Weak<RwLock<InnerDictionary>>,
}

impl InternedString {
    fn new(id: StringId, dictionary: &Arc<RwLock<InnerDictionary>>) -> Self {
        Self {
            id,
            dictionary: Arc::downgrade(dictionary),
        }
    }

    /// Get the string ID
    pub fn id(&self) -> StringId {
        self.id
    }

    /// Get the string value (if dictionary still exists)
    pub fn value(&self) -> Option<String> {
        if let Some(dict) = self.dictionary.upgrade() {
            let dict = dict.read().unwrap();
            dict.entries.get(&self.id).map(|entry| entry.value.clone())
        } else {
            None
        }
    }

    /// Get a reference to the string value
    pub fn as_str(&self) -> Option<String> {
        self.value()
    }
}

impl Drop for InternedString {
    fn drop(&mut self) {
        if let Some(dict) = self.dictionary.upgrade() {
            let mut dict = dict.write().unwrap();
            if let Some(entry) = dict.entries.get_mut(&self.id) {
                if entry.decrement_ref() {
                    // Entry has zero references, mark for potential GC
                    dict.zero_ref_entries.insert(self.id);
                }
            }
        }
    }
}

impl Clone for InternedString {
    fn clone(&self) -> Self {
        if let Some(dict) = self.dictionary.upgrade() {
            let mut dict = dict.write().unwrap();
            if let Some(entry) = dict.entries.get_mut(&self.id) {
                entry.increment_ref();
                dict.zero_ref_entries.remove(&self.id);
            }
        }

        Self {
            id: self.id,
            dictionary: self.dictionary.clone(),
        }
    }
}

impl PartialEq for InternedString {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for InternedString {}

impl std::hash::Hash for InternedString {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl std::fmt::Debug for InternedString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "InternedString(id: {}, value: {:?})",
            self.id,
            self.value()
        )
    }
}

impl std::fmt::Display for InternedString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(value) = self.value() {
            write!(f, "{}", value)
        } else {
            write!(f, "<invalid:{}>", self.id)
        }
    }
}

/// Internal dictionary state
struct InnerDictionary {
    /// Map from string to ID
    string_to_id: HashMap<String, StringId>,
    /// Map from ID to entry
    entries: HashMap<StringId, DictionaryEntry>,
    /// Set of entry IDs with zero references
    zero_ref_entries: HashSet<StringId>,
    /// Next available ID
    next_id: StringId,
    /// Configuration
    config: DictionaryConfig,
    /// Statistics
    stats: DictionaryStats,
    /// Last garbage collection time
    last_gc: Instant,
}

impl InnerDictionary {
    fn new(config: DictionaryConfig) -> Self {
        Self {
            string_to_id: HashMap::new(),
            entries: HashMap::new(),
            zero_ref_entries: HashSet::new(),
            next_id: 1, // Start from 1, reserve 0 for null
            config,
            stats: DictionaryStats::default(),
            last_gc: Instant::now(),
        }
    }

    fn intern_string(&mut self, value: &str) -> StringId {
        self.stats.lookup_count += 1;

        // Check if already interned
        if let Some(&id) = self.string_to_id.get(value) {
            self.stats.hit_count += 1;

            // Update access time and increment reference
            if let Some(entry) = self.entries.get_mut(&id) {
                entry.increment_ref();
                self.zero_ref_entries.remove(&id);
            }

            return id;
        }

        self.stats.miss_count += 1;

        // Create new entry
        let id = self.next_id;
        self.next_id += 1;

        let entry = DictionaryEntry::new(value.to_string());
        let string_len = entry.size;

        self.string_to_id.insert(value.to_string(), id);
        self.entries.insert(id, entry);

        // Update statistics
        self.stats.total_entries += 1;
        self.stats.total_memory_bytes += string_len + std::mem::size_of::<DictionaryEntry>();
        self.stats.active_references += 1;

        if string_len > self.stats.max_string_length {
            self.stats.max_string_length = string_len;
        }

        // Update average string length
        let total_len: usize = self.entries.values().map(|e| e.size).sum();
        self.stats.avg_string_length = total_len as f64 / self.entries.len() as f64;

        // Update load factor
        self.stats.load_factor = self.entries.len() as f64 / self.entries.capacity() as f64;

        // Check if GC is needed
        if self.config.enable_auto_gc && self.should_run_gc() {
            self.run_gc();
        }

        id
    }

    fn get_entry(&self, id: StringId) -> Option<&DictionaryEntry> {
        self.entries.get(&id)
    }

    fn get_string(&self, id: StringId) -> Option<&str> {
        self.entries.get(&id).map(|entry| entry.value.as_str())
    }

    fn should_run_gc(&self) -> bool {
        // Check if enough time has passed
        let time_since_gc = self.last_gc.elapsed();
        if time_since_gc < Duration::from_secs(self.config.gc_interval_seconds) {
            return false;
        }

        // Check various triggers
        self.entries.len() > self.config.max_entries
            || self.stats.total_memory_bytes > self.config.memory_limit_bytes
            || self.stats.load_factor > self.config.load_factor_threshold
            || !self.zero_ref_entries.is_empty()
    }

    fn run_gc(&mut self) {
        let start_time = Instant::now();
        let initial_count = self.entries.len();

        let mut to_remove = Vec::new();

        // Collect entries eligible for removal
        for &id in &self.zero_ref_entries {
            if let Some(entry) = self.entries.get(&id) {
                if entry
                    .is_gc_eligible(self.config.min_gc_age_seconds, self.config.max_idle_seconds)
                {
                    to_remove.push(id);
                }
            }
        }

        // Remove collected entries
        for id in &to_remove {
            if let Some(entry) = self.entries.remove(id) {
                self.string_to_id.remove(&entry.value);
                self.zero_ref_entries.remove(id);

                self.stats.total_memory_bytes = self
                    .stats
                    .total_memory_bytes
                    .saturating_sub(entry.size + std::mem::size_of::<DictionaryEntry>());
            }
        }

        // Update statistics
        self.stats.total_entries = self.entries.len();
        self.stats.gc_runs += 1;
        self.stats.gc_entries_collected += to_remove.len() as u64;
        self.stats.last_gc_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Update load factor
        if self.entries.capacity() > 0 {
            self.stats.load_factor = self.entries.len() as f64 / self.entries.capacity() as f64;
        }

        // Update average string length
        if !self.entries.is_empty() {
            let total_len: usize = self.entries.values().map(|e| e.size).sum();
            self.stats.avg_string_length = total_len as f64 / self.entries.len() as f64;
        }

        self.last_gc = start_time;

        // Shrink hashmaps if they're much larger than needed
        if self.entries.capacity() > self.entries.len() * 4 {
            self.entries.shrink_to_fit();
            self.string_to_id.shrink_to_fit();
        }
    }

    fn force_gc(&mut self) {
        self.run_gc();
    }

    fn clear(&mut self) {
        self.string_to_id.clear();
        self.entries.clear();
        self.zero_ref_entries.clear();
        self.next_id = 1;
        self.stats = DictionaryStats::default();
        self.last_gc = Instant::now();
    }
}

/// String dictionary with automatic interning and garbage collection
pub struct StringDictionary {
    inner: Arc<RwLock<InnerDictionary>>,
}

impl StringDictionary {
    /// Create a new string dictionary
    pub fn new() -> Self {
        Self::with_config(DictionaryConfig::default())
    }

    /// Create a new string dictionary with custom configuration
    pub fn with_config(config: DictionaryConfig) -> Self {
        Self {
            inner: Arc::new(RwLock::new(InnerDictionary::new(config))),
        }
    }

    /// Intern a string and return a handle
    pub fn intern(&self, value: &str) -> InternedString {
        let id = {
            let mut dict = self.inner.write().unwrap();
            dict.intern_string(value)
        };

        InternedString::new(id, &self.inner)
    }

    /// Get a string by ID (without creating a handle)
    pub fn get_string(&self, id: StringId) -> Option<String> {
        let dict = self.inner.read().unwrap();
        dict.get_string(id).map(|s| s.to_string())
    }

    /// Check if a string is already interned
    pub fn contains(&self, value: &str) -> bool {
        let dict = self.inner.read().unwrap();
        dict.string_to_id.contains_key(value)
    }

    /// Get the ID of an interned string (if it exists)
    pub fn get_id(&self, value: &str) -> Option<StringId> {
        let dict = self.inner.read().unwrap();
        dict.string_to_id.get(value).copied()
    }

    /// Force garbage collection
    pub fn gc(&self) {
        let mut dict = self.inner.write().unwrap();
        dict.force_gc();
    }

    /// Get dictionary statistics
    pub fn stats(&self) -> DictionaryStats {
        let dict = self.inner.read().unwrap();
        dict.stats.clone()
    }

    /// Get the number of entries
    pub fn len(&self) -> usize {
        let dict = self.inner.read().unwrap();
        dict.entries.len()
    }

    /// Check if dictionary is empty
    pub fn is_empty(&self) -> bool {
        let dict = self.inner.read().unwrap();
        dict.entries.is_empty()
    }

    /// Clear all entries
    pub fn clear(&self) {
        let mut dict = self.inner.write().unwrap();
        dict.clear();
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let dict = self.inner.read().unwrap();
        dict.stats.total_memory_bytes
    }

    /// Check if memory limit is exceeded
    pub fn is_memory_limit_exceeded(&self) -> bool {
        let dict = self.inner.read().unwrap();
        dict.stats.total_memory_bytes > dict.config.memory_limit_bytes
    }

    /// Validate dictionary integrity
    pub fn validate(&self) -> Result<Vec<String>> {
        let dict = self.inner.read().unwrap();
        let mut issues = Vec::new();

        // Check consistency between maps
        for (string, &id) in &dict.string_to_id {
            if let Some(entry) = dict.entries.get(&id) {
                if entry.value != *string {
                    issues.push(format!(
                        "Inconsistency: string '{}' maps to ID {} but entry has value '{}'",
                        string, id, entry.value
                    ));
                }
            } else {
                issues.push(format!(
                    "Dangling reference: string '{}' maps to non-existent ID {}",
                    string, id
                ));
            }
        }

        // Check reverse mapping
        for (&id, entry) in &dict.entries {
            if let Some(&mapped_id) = dict.string_to_id.get(&entry.value) {
                if mapped_id != id {
                    issues.push(format!(
                        "Reverse mapping inconsistency: entry {} has value '{}' which maps to ID {}",
                        id, entry.value, mapped_id
                    ));
                }
            } else {
                issues.push(format!(
                    "Missing reverse mapping: entry {} with value '{}' not in string_to_id map",
                    id, entry.value
                ));
            }
        }

        // Check zero reference tracking
        for &id in &dict.zero_ref_entries {
            if let Some(entry) = dict.entries.get(&id) {
                if entry.ref_count != 0 {
                    issues.push(format!(
                        "Zero reference tracking error: ID {} has ref_count {} but is in zero_ref_entries",
                        id, entry.ref_count
                    ));
                }
            } else {
                issues.push(format!(
                    "Stale zero reference tracking: ID {} in zero_ref_entries but not in entries",
                    id
                ));
            }
        }

        Ok(issues)
    }
}

impl Default for StringDictionary {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for StringDictionary {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_basic_interning() {
        let dict = StringDictionary::new();

        let s1 = dict.intern("hello");
        let s2 = dict.intern("world");
        let s3 = dict.intern("hello"); // Should reuse

        assert_ne!(s1.id(), s2.id());
        assert_eq!(s1.id(), s3.id());
        assert_eq!(s1.value(), Some("hello".to_string()));
        assert_eq!(s2.value(), Some("world".to_string()));
    }

    #[test]
    fn test_reference_counting() {
        let dict = StringDictionary::new();

        let s1 = dict.intern("test");
        let id = s1.id();
        let s2 = s1.clone(); // Should increment ref count

        // Both handles should work
        assert_eq!(s1.value(), Some("test".to_string()));
        assert_eq!(s2.value(), Some("test".to_string()));

        drop(s1); // Should decrement ref count
        assert_eq!(s2.value(), Some("test".to_string())); // Should still work

        drop(s2); // Should mark for GC

        // String should still exist until GC runs
        assert_eq!(dict.get_string(id), Some("test".to_string()));
    }

    #[test]
    fn test_dictionary_operations() {
        let dict = StringDictionary::new();

        assert!(dict.is_empty());
        assert!(!dict.contains("test"));
        assert_eq!(dict.get_id("test"), None);

        let s1 = dict.intern("test");
        assert!(!dict.is_empty());
        assert!(dict.contains("test"));
        assert_eq!(dict.get_id("test"), Some(s1.id()));
        assert_eq!(dict.len(), 1);

        let _s2 = dict.intern("another");
        assert_eq!(dict.len(), 2);
    }

    #[test]
    fn test_statistics() {
        let dict = StringDictionary::new();

        let _s1 = dict.intern("short");
        let _s2 = dict.intern("much_longer_string");

        let stats = dict.stats();
        assert_eq!(stats.total_entries, 2);
        assert!(stats.total_memory_bytes > 0);
        assert!(stats.avg_string_length > 0.0);
        assert_eq!(stats.max_string_length, "much_longer_string".len());
        assert_eq!(stats.lookup_count, 2);
        assert_eq!(stats.hit_count, 0);
        assert_eq!(stats.miss_count, 2);
    }

    #[test]
    fn test_garbage_collection() {
        let config = DictionaryConfig {
            min_gc_age_seconds: 0, // Allow immediate GC
            max_idle_seconds: 0,   // Allow immediate GC
            enable_auto_gc: false, // Manual GC only
            ..Default::default()
        };

        let dict = StringDictionary::with_config(config);

        // Create and drop a string
        {
            let _s = dict.intern("temp");
        }

        assert_eq!(dict.len(), 1);

        // Force GC
        dict.gc();

        // String should be collected
        assert_eq!(dict.len(), 0);
    }

    #[test]
    fn test_concurrent_access() {
        let dict = Arc::new(StringDictionary::new());
        let mut handles = Vec::new();

        for i in 0..10 {
            let dict_clone = Arc::clone(&dict);
            let handle = thread::spawn(move || {
                let s = dict_clone.intern(&format!("thread_{}", i));
                assert_eq!(s.value(), Some(format!("thread_{}", i)));
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(dict.len(), 10);
    }

    #[test]
    fn test_validation() {
        let dict = StringDictionary::new();

        let _s1 = dict.intern("test1");
        let _s2 = dict.intern("test2");

        let issues = dict.validate().unwrap();
        assert!(issues.is_empty()); // Should be valid
    }

    #[test]
    fn test_memory_management() {
        let config = DictionaryConfig {
            memory_limit_bytes: 1024, // Small limit
            ..Default::default()
        };

        let dict = StringDictionary::with_config(config);

        // Add some strings
        for i in 0..10 {
            let _s = dict.intern(&format!("string_number_{}", i));
        }

        let memory_usage = dict.memory_usage();
        assert!(memory_usage > 0);

        // Check if we can detect memory limit exceeded
        // (depends on string sizes and overhead)
    }

    #[test]
    fn test_clear() {
        let dict = StringDictionary::new();

        let _s1 = dict.intern("test1");
        let _s2 = dict.intern("test2");

        assert_eq!(dict.len(), 2);

        dict.clear();

        assert_eq!(dict.len(), 0);
        assert!(dict.is_empty());

        let stats = dict.stats();
        assert_eq!(stats.total_entries, 0);
        assert_eq!(stats.total_memory_bytes, 0);
    }
}
