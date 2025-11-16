//! Schema-aware autocomplete for SPARQL queries
//!
//! Discovers and caches RDF schema information from datasets to provide
//! intelligent, context-aware autocomplete suggestions in interactive mode.

use crate::cli::completion::{CompletionContext, CompletionItem, CompletionType};
use oxirs_core::model::{Object, Predicate, Subject};
use oxirs_core::rdf_store::RdfStore;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Cache entry for schema information
#[derive(Debug, Clone)]
struct CacheEntry<T> {
    data: T,
    cached_at: Instant,
    ttl: Duration,
}

impl<T> CacheEntry<T> {
    fn new(data: T, ttl: Duration) -> Self {
        Self {
            data,
            cached_at: Instant::now(),
            ttl,
        }
    }

    fn is_expired(&self) -> bool {
        self.cached_at.elapsed() > self.ttl
    }

    fn get(&self) -> Option<&T> {
        if self.is_expired() {
            None
        } else {
            Some(&self.data)
        }
    }
}

/// Schema information discovered from the dataset
#[derive(Debug, Clone, Default)]
pub struct SchemaInfo {
    /// All classes found in the dataset (rdf:type objects)
    pub classes: HashSet<String>,
    /// All properties found in the dataset
    pub properties: HashSet<String>,
    /// Property domain mappings (property -> set of classes)
    pub property_domains: HashMap<String, HashSet<String>>,
    /// Property range mappings (property -> set of classes or datatypes)
    pub property_ranges: HashMap<String, HashSet<String>>,
    /// Frequent property-class combinations (for better suggestions)
    pub property_class_freq: HashMap<(String, String), usize>,
    /// Total number of triples analyzed
    pub triple_count: usize,
}

impl SchemaInfo {
    /// Create a new empty schema info
    pub fn new() -> Self {
        Self::default()
    }

    /// Get classes for a property's domain (classes that have this property)
    pub fn get_domain_classes(&self, property: &str) -> Vec<String> {
        self.property_domains
            .get(property)
            .map(|classes| classes.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Get classes/datatypes for a property's range (valid objects for this property)
    pub fn get_range_types(&self, property: &str) -> Vec<String> {
        self.property_ranges
            .get(property)
            .map(|ranges| ranges.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Get properties commonly used with a class (ranked by frequency)
    pub fn get_class_properties(&self, class: &str) -> Vec<(String, usize)> {
        let mut props: Vec<(String, usize)> = self
            .property_class_freq
            .iter()
            .filter(|((_, c), _)| c == class)
            .map(|((p, _), freq)| (p.clone(), *freq))
            .collect();

        // Sort by frequency (descending)
        props.sort_by(|a, b| b.1.cmp(&a.1));
        props
    }

    /// Get total unique classes discovered
    pub fn class_count(&self) -> usize {
        self.classes.len()
    }

    /// Get total unique properties discovered
    pub fn property_count(&self) -> usize {
        self.properties.len()
    }
}

/// Configuration for schema discovery
#[derive(Debug, Clone)]
pub struct SchemaDiscoveryConfig {
    /// Maximum number of triples to analyze (0 = unlimited)
    pub max_triples: usize,
    /// Cache TTL for schema information
    pub cache_ttl: Duration,
    /// Enable domain/range inference
    pub infer_schema: bool,
    /// Minimum frequency for property-class associations
    pub min_frequency: usize,
}

impl Default for SchemaDiscoveryConfig {
    fn default() -> Self {
        Self {
            max_triples: 100_000,                // Analyze up to 100k triples
            cache_ttl: Duration::from_secs(300), // 5 minute cache
            infer_schema: true,
            min_frequency: 1,
        }
    }
}

impl SchemaDiscoveryConfig {
    /// Create configuration for small datasets (analyze everything)
    pub fn for_small_dataset() -> Self {
        Self {
            max_triples: 0,                      // Unlimited
            cache_ttl: Duration::from_secs(600), // 10 minutes
            infer_schema: true,
            min_frequency: 1,
        }
    }

    /// Create configuration for large datasets (sample)
    pub fn for_large_dataset() -> Self {
        Self {
            max_triples: 50_000,                 // Sample 50k triples
            cache_ttl: Duration::from_secs(180), // 3 minute cache
            infer_schema: false,                 // Skip inference for speed
            min_frequency: 5,                    // Require higher frequency
        }
    }
}

/// Schema-aware autocomplete provider
pub struct SchemaAutocompleteProvider {
    /// RDF store to query for schema information
    store: Arc<RwLock<RdfStore>>,
    /// Cached schema information
    schema_cache: Arc<RwLock<Option<CacheEntry<SchemaInfo>>>>,
    /// Configuration
    config: SchemaDiscoveryConfig,
}

impl SchemaAutocompleteProvider {
    /// Create a new schema autocomplete provider
    pub fn new(store: Arc<RwLock<RdfStore>>) -> Self {
        Self {
            store,
            schema_cache: Arc::new(RwLock::new(None)),
            config: SchemaDiscoveryConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(store: Arc<RwLock<RdfStore>>, config: SchemaDiscoveryConfig) -> Self {
        Self {
            store,
            schema_cache: Arc::new(RwLock::new(None)),
            config,
        }
    }

    /// Discover schema information from the dataset
    pub fn discover_schema(&self) -> Result<SchemaInfo, String> {
        let store = self.store.read().map_err(|e| e.to_string())?;

        let mut schema = SchemaInfo::new();
        let mut triple_count = 0;

        // Get all triples from the store
        let all_triples = store.triples().map_err(|e| e.to_string())?;

        let triples_to_analyze =
            if self.config.max_triples > 0 && all_triples.len() > self.config.max_triples {
                &all_triples[..self.config.max_triples]
            } else {
                &all_triples
            };

        for triple in triples_to_analyze {
            triple_count += 1;

            // Extract strings from terms
            let _subject_str = Self::subject_to_string(triple.subject());
            let predicate_str = Self::predicate_to_string(triple.predicate());
            let object_str = Self::object_to_string(triple.object());

            // Track all properties
            schema.properties.insert(predicate_str.clone());

            // Special handling for rdf:type to discover classes
            if predicate_str == "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"
                || predicate_str == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
                || predicate_str == "rdf:type"
            {
                schema.classes.insert(object_str.clone());

                // Track property-class associations
                let key = (predicate_str.clone(), object_str.clone());
                *schema.property_class_freq.entry(key).or_insert(0) += 1;
            }

            // Infer schema if enabled
            if self.config.infer_schema {
                // Track property domains (which classes have this property)
                // This would require more sophisticated analysis in production

                // Track property ranges (what types of objects this property has)
                if !object_str.is_empty() {
                    schema
                        .property_ranges
                        .entry(predicate_str.clone())
                        .or_default()
                        .insert(Self::infer_type(&object_str));
                }
            }
        }

        schema.triple_count = triple_count;
        Ok(schema)
    }

    /// Get cached schema or discover it
    pub fn get_schema(&self) -> Result<SchemaInfo, String> {
        // Check cache first
        {
            let cache = self.schema_cache.read().map_err(|e| e.to_string())?;
            if let Some(entry) = cache.as_ref() {
                if let Some(schema) = entry.get() {
                    return Ok(schema.clone());
                }
            }
        }

        // Cache miss or expired, discover schema
        let schema = self.discover_schema()?;

        // Update cache
        {
            let mut cache = self.schema_cache.write().map_err(|e| e.to_string())?;
            *cache = Some(CacheEntry::new(schema.clone(), self.config.cache_ttl));
        }

        Ok(schema)
    }

    /// Invalidate the schema cache (call when dataset changes)
    pub fn invalidate_cache(&self) {
        if let Ok(mut cache) = self.schema_cache.write() {
            *cache = None;
        }
    }

    /// Get class name suggestions
    pub fn suggest_classes(&self, prefix: &str) -> Result<Vec<CompletionItem>, String> {
        let schema = self.get_schema()?;

        let prefix_lower = prefix.to_lowercase();
        let mut suggestions: Vec<CompletionItem> = schema
            .classes
            .iter()
            .filter(|class| class.to_lowercase().contains(&prefix_lower))
            .map(|class| CompletionItem {
                replacement: class.clone(),
                display: Self::extract_local_name(class),
                description: Some("Class".to_string()),
                completion_type: CompletionType::Value,
            })
            .collect();

        // Sort by relevance (exact prefix match first, then alphabetically)
        suggestions.sort_by(|a, b| {
            let a_starts = a.replacement.to_lowercase().starts_with(&prefix_lower);
            let b_starts = b.replacement.to_lowercase().starts_with(&prefix_lower);

            match (a_starts, b_starts) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => a.replacement.cmp(&b.replacement),
            }
        });

        Ok(suggestions)
    }

    /// Get property name suggestions
    pub fn suggest_properties(
        &self,
        prefix: &str,
        context_class: Option<&str>,
    ) -> Result<Vec<CompletionItem>, String> {
        let schema = self.get_schema()?;

        let prefix_lower = prefix.to_lowercase();

        // If we have context (subject class), prioritize properties used with that class
        let mut suggestions: Vec<(String, usize)> = if let Some(class) = context_class {
            schema.get_class_properties(class)
        } else {
            schema.properties.iter().map(|p| (p.clone(), 1)).collect()
        };

        // Filter by prefix
        suggestions.retain(|(prop, _)| prop.to_lowercase().contains(&prefix_lower));

        // Sort by frequency and relevance
        suggestions.sort_by(|a, b| {
            let a_starts = a.0.to_lowercase().starts_with(&prefix_lower);
            let b_starts = b.0.to_lowercase().starts_with(&prefix_lower);

            match (a_starts, b_starts) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => b.1.cmp(&a.1).then(a.0.cmp(&b.0)),
            }
        });

        let items: Vec<CompletionItem> = suggestions
            .into_iter()
            .map(|(prop, freq)| CompletionItem {
                replacement: prop.clone(),
                display: Self::extract_local_name(&prop),
                description: Some(format!("Property (used {} times)", freq)),
                completion_type: CompletionType::Variable,
            })
            .collect();

        Ok(items)
    }

    /// Get context-aware suggestions based on query position
    pub fn get_contextual_suggestions(
        &self,
        context: &CompletionContext,
    ) -> Result<Vec<CompletionItem>, String> {
        let prefix = &context.current_word;

        // Determine what type of suggestion is appropriate based on context
        match self.determine_suggestion_type(context) {
            SuggestionType::Class => self.suggest_classes(prefix),
            SuggestionType::Property => self.suggest_properties(prefix, None),
            SuggestionType::PropertyWithContext(class) => {
                self.suggest_properties(prefix, Some(&class))
            }
            SuggestionType::None => Ok(Vec::new()),
        }
    }

    /// Determine what type of suggestion to provide based on context
    fn determine_suggestion_type(&self, context: &CompletionContext) -> SuggestionType {
        // Reconstruct the line from context
        let line_parts: Vec<String> = context.args.clone();
        let line = line_parts.join(" ");
        let pos = line.len();

        // Look at the text before the cursor
        let before_cursor = &line[..pos];

        // Check for rdf:type or "a" (which means rdf:type in SPARQL)
        if before_cursor.contains("rdf:type") || before_cursor.ends_with(" a ") {
            return SuggestionType::Class;
        }

        // Check if we're in a WHERE clause after subject
        if before_cursor.contains("WHERE") && before_cursor.contains("?") {
            // Check if we're at a property position (after subject, before object)
            let tokens: Vec<&str> = before_cursor.split_whitespace().collect();
            if tokens.len() >= 2 {
                let last_token = tokens[tokens.len() - 1];
                let second_last = tokens[tokens.len() - 2];

                if second_last.starts_with('?') && !last_token.starts_with('?') {
                    // We're at property position
                    return SuggestionType::Property;
                }
            }
        }

        SuggestionType::None
    }

    /// Convert a Subject to a string representation
    fn subject_to_string(subject: &Subject) -> String {
        match subject {
            Subject::NamedNode(node) => node.to_string(),
            Subject::BlankNode(node) => format!("_:{}", node),
            Subject::Variable(var) => format!("?{}", var),
            Subject::QuotedTriple(_) => String::new(), // Skip for schema discovery
        }
    }

    /// Convert a Predicate to a string representation
    fn predicate_to_string(predicate: &Predicate) -> String {
        match predicate {
            Predicate::NamedNode(node) => node.to_string(),
            Predicate::Variable(var) => format!("?{}", var),
        }
    }

    /// Convert an Object to a string representation
    fn object_to_string(obj: &Object) -> String {
        match obj {
            Object::NamedNode(node) => node.to_string(),
            Object::BlankNode(node) => format!("_:{}", node),
            Object::Literal(lit) => lit.value().to_string(),
            Object::Variable(var) => format!("?{}", var),
            Object::QuotedTriple(_) => String::new(), // Skip for schema discovery
        }
    }

    /// Infer the type of a value (simplified version)
    fn infer_type(value: &str) -> String {
        if value.starts_with("http://") || value.starts_with("https://") {
            "IRI".to_string()
        } else if value.parse::<i64>().is_ok() {
            "xsd:integer".to_string()
        } else if value.parse::<f64>().is_ok() {
            "xsd:decimal".to_string()
        } else if value == "true" || value == "false" {
            "xsd:boolean".to_string()
        } else {
            "xsd:string".to_string()
        }
    }

    /// Extract local name from a URI (e.g., "foaf:Person" from "http://xmlns.com/foaf/0.1/Person")
    fn extract_local_name(uri: &str) -> String {
        if let Some(pos) = uri.rfind(&['/', '#'][..]) {
            uri[pos + 1..].to_string()
        } else {
            uri.to_string()
        }
    }

    /// Get statistics about the schema cache
    pub fn get_cache_stats(&self) -> Result<CacheStats, String> {
        let cache = self.schema_cache.read().map_err(|e| e.to_string())?;

        if let Some(entry) = cache.as_ref() {
            if let Some(schema) = entry.get() {
                return Ok(CacheStats {
                    is_cached: true,
                    is_expired: false,
                    cached_at: Some(entry.cached_at),
                    class_count: schema.class_count(),
                    property_count: schema.property_count(),
                    triple_count: schema.triple_count,
                });
            } else {
                return Ok(CacheStats {
                    is_cached: true,
                    is_expired: true,
                    cached_at: Some(entry.cached_at),
                    class_count: 0,
                    property_count: 0,
                    triple_count: 0,
                });
            }
        }

        Ok(CacheStats {
            is_cached: false,
            is_expired: false,
            cached_at: None,
            class_count: 0,
            property_count: 0,
            triple_count: 0,
        })
    }
}

/// Type of suggestion to provide
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
enum SuggestionType {
    Class,
    Property,
    PropertyWithContext(String),
    None,
}

/// Statistics about the schema cache
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub is_cached: bool,
    pub is_expired: bool,
    pub cached_at: Option<Instant>,
    pub class_count: usize,
    pub property_count: usize,
    pub triple_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_core::model::{Literal, NamedNode, Object, Predicate, Subject};

    fn create_test_store() -> Arc<RwLock<RdfStore>> {
        let mut store = RdfStore::new().expect("Failed to create store");

        // Add some test triples
        let rdf_type = Predicate::NamedNode(
            NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").unwrap(),
        );
        let foaf_person =
            Object::NamedNode(NamedNode::new("http://xmlns.com/foaf/0.1/Person").unwrap());
        let foaf_name =
            Predicate::NamedNode(NamedNode::new("http://xmlns.com/foaf/0.1/name").unwrap());

        // Person instance
        let alice = Subject::NamedNode(NamedNode::new("http://example.org/alice").unwrap());

        // Use the Triple API instead
        use oxirs_core::model::Triple;
        let triple1 = Triple::new(alice.clone(), rdf_type.clone(), foaf_person.clone());
        let triple2 = Triple::new(
            alice.clone(),
            foaf_name.clone(),
            Object::Literal(Literal::new_simple_literal("Alice")),
        );

        store.insert_triple(triple1).ok();
        store.insert_triple(triple2).ok();

        Arc::new(RwLock::new(store))
    }

    #[test]
    fn test_schema_discovery() {
        let store = create_test_store();
        let provider = SchemaAutocompleteProvider::new(store);

        let schema = provider.discover_schema().unwrap();

        // Debug output
        eprintln!("Triple count: {}", schema.triple_count);
        eprintln!("Classes found: {:?}", schema.classes);
        eprintln!("Properties found: {:?}", schema.properties);

        assert!(schema.class_count() > 0, "No classes discovered");
        assert!(schema.property_count() > 0, "No properties discovered");
        assert!(schema.triple_count > 0, "No triples found");
    }

    #[test]
    fn test_class_discovery() {
        let store = create_test_store();
        let provider = SchemaAutocompleteProvider::new(store);

        let schema = provider.get_schema().unwrap();

        // Should discover foaf:Person
        assert!(schema.classes.iter().any(|c| c.contains("Person")));
    }

    #[test]
    fn test_property_discovery() {
        let store = create_test_store();
        let provider = SchemaAutocompleteProvider::new(store);

        let schema = provider.get_schema().unwrap();

        // Should discover foaf:name and rdf:type
        assert!(schema.properties.iter().any(|p| p.contains("name")));
        assert!(schema.properties.iter().any(|p| p.contains("type")));
    }

    #[test]
    fn test_class_suggestions() {
        let store = create_test_store();
        let provider = SchemaAutocompleteProvider::new(store);

        let suggestions = provider.suggest_classes("Per").unwrap();

        assert!(!suggestions.is_empty());
        assert!(suggestions.iter().any(|s| s.display.contains("Person")));
    }

    #[test]
    fn test_property_suggestions() {
        let store = create_test_store();
        let provider = SchemaAutocompleteProvider::new(store);

        let suggestions = provider.suggest_properties("name", None).unwrap();

        assert!(!suggestions.is_empty());
        assert!(suggestions.iter().any(|s| s.display.contains("name")));
    }

    #[test]
    fn test_cache_functionality() {
        let store = create_test_store();
        let provider = SchemaAutocompleteProvider::new(store);

        // First call should populate cache
        let schema1 = provider.get_schema().unwrap();
        let stats1 = provider.get_cache_stats().unwrap();
        assert!(stats1.is_cached);

        // Second call should use cache
        let schema2 = provider.get_schema().unwrap();
        assert_eq!(schema1.class_count(), schema2.class_count());

        // Invalidate cache
        provider.invalidate_cache();
        let stats2 = provider.get_cache_stats().unwrap();
        assert!(!stats2.is_cached);
    }

    #[test]
    fn test_cache_expiration() {
        let store = create_test_store();
        let config = SchemaDiscoveryConfig {
            cache_ttl: Duration::from_millis(10), // Very short TTL for testing
            ..Default::default()
        };
        let provider = SchemaAutocompleteProvider::with_config(store, config);

        // Populate cache
        provider.get_schema().ok();

        // Wait for expiration
        std::thread::sleep(Duration::from_millis(20));

        // Cache should be expired but present
        let stats = provider.get_cache_stats().unwrap();
        assert!(stats.is_cached);
        assert!(stats.is_expired);
    }

    #[test]
    fn test_empty_store() {
        let empty_store = Arc::new(RwLock::new(
            RdfStore::new().expect("Failed to create store"),
        ));
        let provider = SchemaAutocompleteProvider::new(empty_store);

        let schema = provider.get_schema().unwrap();

        assert_eq!(schema.class_count(), 0);
        assert_eq!(schema.property_count(), 0);
    }

    #[test]
    fn test_local_name_extraction() {
        assert_eq!(
            SchemaAutocompleteProvider::extract_local_name("http://xmlns.com/foaf/0.1/Person"),
            "Person"
        );
        assert_eq!(
            SchemaAutocompleteProvider::extract_local_name("http://example.org#name"),
            "name"
        );
        assert_eq!(
            SchemaAutocompleteProvider::extract_local_name("simple"),
            "simple"
        );
    }

    #[test]
    fn test_type_inference() {
        assert_eq!(
            SchemaAutocompleteProvider::infer_type("http://example.org"),
            "IRI"
        );
        assert_eq!(SchemaAutocompleteProvider::infer_type("42"), "xsd:integer");
        assert_eq!(
            SchemaAutocompleteProvider::infer_type("3.14"),
            "xsd:decimal"
        );
        assert_eq!(
            SchemaAutocompleteProvider::infer_type("true"),
            "xsd:boolean"
        );
        assert_eq!(
            SchemaAutocompleteProvider::infer_type("hello"),
            "xsd:string"
        );
    }

    #[test]
    fn test_small_dataset_config() {
        let config = SchemaDiscoveryConfig::for_small_dataset();
        assert_eq!(config.max_triples, 0); // Unlimited
        assert!(config.infer_schema);
    }

    #[test]
    fn test_large_dataset_config() {
        let config = SchemaDiscoveryConfig::for_large_dataset();
        assert_eq!(config.max_triples, 50_000);
        assert!(!config.infer_schema); // Disabled for performance
    }
}
