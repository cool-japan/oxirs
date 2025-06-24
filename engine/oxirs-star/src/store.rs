//! RDF-star storage implementation with efficient handling of quoted triples.
//!
//! This module provides storage backends for RDF-star data, extending the core
//! OxiRS storage with support for quoted triples and efficient indexing.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use oxirs_core::store::Store as CoreStore;
use tracing::{debug, info, span, Level};

use crate::model::{StarGraph, StarQuad, StarTerm, StarTriple};
use crate::{StarConfig, StarError, StarResult, StarStatistics};

/// RDF-star storage backend with support for quoted triples
#[derive(Debug, Clone)]
pub struct StarStore {
    /// Core RDF storage backend
    core_store: Arc<RwLock<CoreStore>>,
    /// RDF-star specific triples (those containing quoted triples)
    star_triples: Arc<RwLock<Vec<StarTriple>>>,
    /// Quoted triple index for efficient lookup
    quoted_triple_index: Arc<RwLock<HashMap<String, HashSet<usize>>>>,
    /// Configuration for the store
    config: StarConfig,
    /// Statistics tracking
    statistics: Arc<RwLock<StarStatistics>>,
}

impl StarStore {
    /// Create a new RDF-star store with default configuration
    pub fn new() -> Self {
        Self::with_config(StarConfig::default())
    }

    /// Create a new RDF-star store with custom configuration
    pub fn with_config(config: StarConfig) -> Self {
        let span = span!(Level::INFO, "new_star_store");
        let _enter = span.enter();

        info!("Creating new RDF-star store");
        debug!("Configuration: {:?}", config);

        Self {
            core_store: Arc::new(RwLock::new(CoreStore::new().expect("Failed to create core store"))),
            star_triples: Arc::new(RwLock::new(Vec::new())),
            quoted_triple_index: Arc::new(RwLock::new(HashMap::new())),
            config,
            statistics: Arc::new(RwLock::new(StarStatistics::default())),
        }
    }

    /// Insert a RDF-star triple into the store
    pub fn insert(&self, triple: &StarTriple) -> StarResult<()> {
        let span = span!(Level::DEBUG, "insert_triple");
        let _enter = span.enter();

        let start_time = Instant::now();

        // Validate the triple
        triple.validate()?;

        // Check nesting depth
        crate::validate_nesting_depth(&triple.subject, self.config.max_nesting_depth)?;
        crate::validate_nesting_depth(&triple.predicate, self.config.max_nesting_depth)?;
        crate::validate_nesting_depth(&triple.object, self.config.max_nesting_depth)?;

        // Insert into appropriate storage
        if triple.contains_quoted_triples() {
            self.insert_star_triple(triple)?;
        } else {
            self.insert_regular_triple(triple)?;
        }

        // Update statistics
        {
            let mut stats = self.statistics.write().unwrap();
            stats.processing_time_us += start_time.elapsed().as_micros() as u64;
            if triple.contains_quoted_triples() {
                stats.quoted_triples_count += 1;
                stats.max_nesting_encountered = stats.max_nesting_encountered.max(triple.nesting_depth());
            }
        }

        debug!("Inserted triple: {}", triple);
        Ok(())
    }

    /// Insert a regular RDF triple (no quoted triples) into core store
    fn insert_regular_triple(&self, triple: &StarTriple) -> StarResult<()> {
        // Convert to core RDF format and insert
        // This is a simplified conversion - in reality we'd need proper conversion
        // from StarTerm to core RDF terms
        debug!("Inserting regular triple into core store");
        
        // For now, we'll store all triples in the star storage
        // In a full implementation, we'd convert non-star triples to core format
        let mut star_triples = self.star_triples.write().unwrap();
        star_triples.push(triple.clone());
        
        Ok(())
    }

    /// Insert a RDF-star triple (containing quoted triples) into star storage
    fn insert_star_triple(&self, triple: &StarTriple) -> StarResult<()> {
        let mut star_triples = self.star_triples.write().unwrap();
        let mut index = self.quoted_triple_index.write().unwrap();

        let triple_index = star_triples.len();
        star_triples.push(triple.clone());

        // Build index for quoted triples
        self.index_quoted_triples(triple, triple_index, &mut index);

        debug!("Inserted star triple with {} quoted triples", self.count_quoted_triples_in_triple(triple));
        Ok(())
    }

    /// Build index entries for quoted triples in a given triple
    fn index_quoted_triples(&self, triple: &StarTriple, triple_index: usize, index: &mut HashMap<String, HashSet<usize>>) {
        // Index quoted triples in subject
        if let StarTerm::QuotedTriple(qt) = &triple.subject {
            let key = self.quoted_triple_key(qt);
            index.entry(key).or_insert_with(HashSet::new).insert(triple_index);
            self.index_quoted_triples(qt, triple_index, index);
        }

        // Index quoted triples in object
        if let StarTerm::QuotedTriple(qt) = &triple.object {
            let key = self.quoted_triple_key(qt);
            index.entry(key).or_insert_with(HashSet::new).insert(triple_index);
            self.index_quoted_triples(qt, triple_index, index);
        }
    }

    /// Generate a key for indexing quoted triples
    fn quoted_triple_key(&self, triple: &StarTriple) -> String {
        format!("{}|{}|{}", triple.subject, triple.predicate, triple.object)
    }

    /// Count quoted triples within a single triple
    fn count_quoted_triples_in_triple(&self, triple: &StarTriple) -> usize {
        let mut count = 0;
        
        if triple.subject.is_quoted_triple() {
            count += 1;
            if let StarTerm::QuotedTriple(qt) = &triple.subject {
                count += self.count_quoted_triples_in_triple(qt);
            }
        }
        
        if triple.predicate.is_quoted_triple() {
            count += 1;
            if let StarTerm::QuotedTriple(qt) = &triple.predicate {
                count += self.count_quoted_triples_in_triple(qt);
            }
        }
        
        if triple.object.is_quoted_triple() {
            count += 1;
            if let StarTerm::QuotedTriple(qt) = &triple.object {
                count += self.count_quoted_triples_in_triple(qt);
            }
        }
        
        count
    }

    /// Remove a triple from the store
    pub fn remove(&self, triple: &StarTriple) -> StarResult<bool> {
        let span = span!(Level::DEBUG, "remove_triple");
        let _enter = span.enter();

        let mut star_triples = self.star_triples.write().unwrap();
        
        if let Some(pos) = star_triples.iter().position(|t| t == triple) {
            star_triples.remove(pos);
            
            // Update index
            let mut index = self.quoted_triple_index.write().unwrap();
            for (_, indices) in index.iter_mut() {
                indices.remove(&pos);
                // Shift indices down for elements after the removed position
                let indices_to_update: Vec<usize> = indices.iter().filter(|&&i| i > pos).cloned().collect();
                for idx in indices_to_update {
                    indices.remove(&idx);
                    indices.insert(idx - 1);
                }
            }
            
            debug!("Removed triple: {}", triple);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Check if the store contains a specific triple
    pub fn contains(&self, triple: &StarTriple) -> bool {
        let star_triples = self.star_triples.read().unwrap();
        star_triples.contains(triple)
    }

    /// Get all triples in the store
    pub fn triples(&self) -> Vec<StarTriple> {
        let star_triples = self.star_triples.read().unwrap();
        star_triples.clone()
    }

    /// Query for triples matching a pattern
    pub fn query_triples(&self, subject: Option<&StarTerm>, predicate: Option<&StarTerm>, object: Option<&StarTerm>) -> Vec<StarTriple> {
        let span = span!(Level::DEBUG, "query_triples");
        let _enter = span.enter();

        let star_triples = self.star_triples.read().unwrap();
        
        star_triples
            .iter()
            .filter(|triple| {
                let subject_matches = subject.map_or(true, |s| &triple.subject == s);
                let predicate_matches = predicate.map_or(true, |p| &triple.predicate == p);
                let object_matches = object.map_or(true, |o| &triple.object == o);
                
                subject_matches && predicate_matches && object_matches
            })
            .cloned()
            .collect()
    }

    /// Find triples that contain a specific quoted triple
    pub fn find_triples_containing_quoted(&self, quoted_triple: &StarTriple) -> Vec<StarTriple> {
        let span = span!(Level::DEBUG, "find_triples_containing_quoted");
        let _enter = span.enter();

        let key = self.quoted_triple_key(quoted_triple);
        let index = self.quoted_triple_index.read().unwrap();
        let star_triples = self.star_triples.read().unwrap();

        if let Some(indices) = index.get(&key) {
            indices
                .iter()
                .filter_map(|&idx| star_triples.get(idx))
                .cloned()
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get the number of triples in the store
    pub fn len(&self) -> usize {
        let star_triples = self.star_triples.read().unwrap();
        star_triples.len()
    }

    /// Check if the store is empty
    pub fn is_empty(&self) -> bool {
        let star_triples = self.star_triples.read().unwrap();
        star_triples.is_empty()
    }

    /// Clear all triples from the store
    pub fn clear(&self) -> StarResult<()> {
        let span = span!(Level::INFO, "clear_store");
        let _enter = span.enter();

        {
            let mut star_triples = self.star_triples.write().unwrap();
            star_triples.clear();
        }

        {
            let mut index = self.quoted_triple_index.write().unwrap();
            index.clear();
        }

        {
            let mut stats = self.statistics.write().unwrap();
            *stats = StarStatistics::default();
        }

        info!("Cleared all triples from store");
        Ok(())
    }

    /// Get statistics about the store
    pub fn statistics(&self) -> StarStatistics {
        let stats = self.statistics.read().unwrap();
        stats.clone()
    }

    /// Export the store as a StarGraph
    pub fn to_graph(&self) -> StarGraph {
        let star_triples = self.star_triples.read().unwrap();
        let mut graph = StarGraph::new();
        
        for triple in star_triples.iter() {
            // Unwrap is safe here because we validate on insert
            graph.insert(triple.clone()).unwrap();
        }
        
        graph
    }

    /// Import triples from a StarGraph
    pub fn from_graph(&self, graph: &StarGraph) -> StarResult<()> {
        let span = span!(Level::INFO, "import_from_graph");
        let _enter = span.enter();

        for triple in graph.triples() {
            self.insert(triple)?;
        }

        info!("Imported {} triples from graph", graph.len());
        Ok(())
    }

    /// Optimize the store by rebuilding indices
    pub fn optimize(&self) -> StarResult<()> {
        let span = span!(Level::INFO, "optimize_store");
        let _enter = span.enter();

        let star_triples = self.star_triples.read().unwrap();
        let mut index = self.quoted_triple_index.write().unwrap();
        
        // Rebuild the quoted triple index
        index.clear();
        for (i, triple) in star_triples.iter().enumerate() {
            if triple.contains_quoted_triples() {
                self.index_quoted_triples(triple, i, &mut index);
            }
        }

        info!("Store optimization completed");
        Ok(())
    }

    /// Get configuration
    pub fn config(&self) -> &StarConfig {
        &self.config
    }

    /// Update configuration (requires store recreation for some settings)
    pub fn update_config(&mut self, config: StarConfig) -> StarResult<()> {
        // Validate new configuration
        crate::init_star_system(config.clone())?;
        self.config = config;
        Ok(())
    }
}

impl Default for StarStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator over triples in the store
pub struct StarTripleIterator<'a> {
    triples: std::slice::Iter<'a, StarTriple>,
}

impl<'a> Iterator for StarTripleIterator<'a> {
    type Item = &'a StarTriple;

    fn next(&mut self) -> Option<Self::Item> {
        self.triples.next()
    }
}

impl StarStore {
    /// Get an iterator over all triples
    pub fn iter(&self) -> StarTripleIterator {
        let star_triples = self.star_triples.read().unwrap();
        // Note: This is a simplified implementation
        // In practice, we'd need a more sophisticated iterator that doesn't hold the lock
        StarTripleIterator {
            triples: unsafe { std::mem::transmute(star_triples.iter()) },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::StarTerm;

    #[test]
    fn test_store_creation() {
        let store = StarStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_basic_operations() {
        let store = StarStore::new();
        
        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/knows").unwrap(),
            StarTerm::iri("http://example.org/bob").unwrap(),
        );

        // Insert
        store.insert(&triple).unwrap();
        assert_eq!(store.len(), 1);
        assert!(store.contains(&triple));

        // Query
        let results = store.query_triples(
            Some(&StarTerm::iri("http://example.org/alice").unwrap()),
            None,
            None,
        );
        assert_eq!(results.len(), 1);

        // Remove
        assert!(store.remove(&triple).unwrap());
        assert!(store.is_empty());
    }

    #[test]
    fn test_quoted_triple_operations() {
        let store = StarStore::new();

        // Create a quoted triple
        let inner = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/age").unwrap(),
            StarTerm::literal("25").unwrap(),
        );

        let outer = StarTriple::new(
            StarTerm::quoted_triple(inner.clone()),
            StarTerm::iri("http://example.org/certainty").unwrap(),
            StarTerm::literal("0.9").unwrap(),
        );

        store.insert(&outer).unwrap();
        assert_eq!(store.len(), 1);

        // Find triples containing the quoted triple
        let containing = store.find_triples_containing_quoted(&inner);
        assert_eq!(containing.len(), 1);
        assert_eq!(containing[0], outer);
    }

    #[test]
    fn test_store_statistics() {
        let store = StarStore::new();
        
        let regular = StarTriple::new(
            StarTerm::iri("http://example.org/s").unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::iri("http://example.org/o").unwrap(),
        );

        let quoted = StarTriple::new(
            StarTerm::quoted_triple(regular.clone()),
            StarTerm::iri("http://example.org/certainty").unwrap(),
            StarTerm::literal("high").unwrap(),
        );

        store.insert(&regular).unwrap();
        store.insert(&quoted).unwrap();

        let stats = store.statistics();
        assert_eq!(stats.quoted_triples_count, 1);
        assert_eq!(stats.max_nesting_encountered, 1);
    }

    #[test]
    fn test_graph_import_export() {
        let store = StarStore::new();
        let mut graph = StarGraph::new();

        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/s").unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::iri("http://example.org/o").unwrap(),
        );

        graph.insert(triple.clone()).unwrap();
        store.from_graph(&graph).unwrap();

        assert_eq!(store.len(), 1);
        assert!(store.contains(&triple));

        let exported = store.to_graph();
        assert_eq!(exported.len(), 1);
        assert!(exported.contains(&triple));
    }
}