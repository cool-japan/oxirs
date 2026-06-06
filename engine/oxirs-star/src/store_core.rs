//! Core RDF-star store implementation.
//!
//! This sibling module of `crate::store` defines the [`StarStore`] struct,
//! its internal state helpers, basic constructor / mutator operations, and the
//! conversion helpers that turn [`StarTerm`] values into core RDF terms.

use std::sync::{Arc, RwLock};
use std::time::Instant;

use oxirs_core::rdf_store::{ConcreteStore as CoreStore, Store};
use tracing::{debug, info, span, Level};

use crate::model::{StarTerm, StarTriple};
use crate::store::{cache_mod, conversion, index};
use crate::{StarConfig, StarError, StarResult, StarStatistics};

use cache_mod::{CacheConfig, StarCache};
use index::QuotedTripleIndex;

/// RDF-star storage backend with support for quoted triples
#[derive(Clone)]
pub struct StarStore {
    /// Core RDF storage backend
    pub(crate) core_store: Arc<RwLock<CoreStore>>,
    /// RDF-star specific triples (those containing quoted triples)
    pub(crate) star_triples: Arc<RwLock<Vec<StarTriple>>>,
    /// Enhanced B-tree based quoted triple index for efficient lookup
    pub(crate) quoted_triple_index: Arc<RwLock<QuotedTripleIndex>>,
    /// Configuration for the store
    pub(crate) config: StarConfig,
    /// Statistics tracking
    pub(crate) statistics: Arc<RwLock<StarStatistics>>,
    /// Cache for frequently accessed data
    pub(crate) cache: Arc<StarCache>,
    /// Bulk insertion state
    pub(crate) bulk_insert_state: Arc<RwLock<BulkInsertState>>,
    /// Memory-mapped storage state
    pub(crate) memory_mapped: Arc<RwLock<MemoryMappedState>>,
}

/// State tracking for bulk insertion operations
#[derive(Debug, Default)]
pub(crate) struct BulkInsertState {
    /// Whether bulk insertion is currently active
    pub(crate) active: bool,
    /// Pending triples waiting to be indexed
    pub(crate) pending_triples: Vec<StarTriple>,
    /// Memory usage tracking for bulk operations
    pub(crate) current_memory_usage: usize,
    /// Batch count for monitoring
    pub(crate) batch_count: usize,
}

/// State for memory-mapped storage operations
#[derive(Debug, Default)]
pub(crate) struct MemoryMappedState {
    /// Whether memory mapping is enabled
    pub(crate) enabled: bool,
    /// Path to the memory-mapped file
    pub(crate) file_path: Option<String>,
    /// Compression settings for stored data
    pub(crate) compression_enabled: bool,
    /// Last sync timestamp
    pub(crate) last_sync: Option<Instant>,
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

        info!("Creating new RDF-star store with optimizations");
        debug!("Configuration: {:?}", config);

        Self {
            core_store: Arc::new(RwLock::new(
                CoreStore::new().expect("Failed to create core store"),
            )),
            star_triples: Arc::new(RwLock::new(Vec::new())),
            quoted_triple_index: Arc::new(RwLock::new(QuotedTripleIndex::new())),
            config: config.clone(),
            statistics: Arc::new(RwLock::new(StarStatistics::default())),
            cache: Arc::new(StarCache::new(CacheConfig::default())),
            bulk_insert_state: Arc::new(RwLock::new(BulkInsertState::default())),
            memory_mapped: Arc::new(RwLock::new(MemoryMappedState::default())),
        }
    }

    /// Get the store configuration
    pub fn config(&self) -> &StarConfig {
        &self.config
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
        eprintln!(
            "DEBUG INSERT: Triple contains quoted triples: {}",
            triple.contains_quoted_triples()
        );
        if triple.contains_quoted_triples() {
            eprintln!("DEBUG INSERT: Inserting as star triple");
            self.insert_star_triple(triple)?;
        } else {
            eprintln!("DEBUG INSERT: Inserting as regular triple");
            self.insert_regular_triple(triple)?;
        }

        // Update statistics
        {
            let mut stats = self.statistics.write().unwrap_or_else(|e| e.into_inner());
            stats.processing_time_us += start_time.elapsed().as_micros() as u64;
            if triple.contains_quoted_triples() {
                stats.quoted_triples_count += 1;
                stats.max_nesting_encountered =
                    stats.max_nesting_encountered.max(triple.nesting_depth());
            }
        }

        debug!("Inserted triple: {}", triple);
        Ok(())
    }

    /// Insert a regular RDF triple (no quoted triples) into core store
    pub(crate) fn insert_regular_triple(&self, triple: &StarTriple) -> StarResult<()> {
        eprintln!("DEBUG: Inserting regular triple into core store");

        // Convert StarTriple to core RDF triple
        let core_triple = self.convert_to_core_triple(triple)?;
        eprintln!("DEBUG: Converted to core triple successfully");

        // Insert into core store (convert triple to quad in default graph)
        let core_quad = oxirs_core::model::Quad::from_triple(core_triple);
        eprintln!("DEBUG: Created core quad for insertion: {core_quad:?}");
        let core_store = self.core_store.write().unwrap_or_else(|e| e.into_inner());
        let result = CoreStore::insert_quad(&core_store, core_quad).map_err(StarError::CoreError);
        eprintln!("DEBUG: Core store insert result: {result:?}");
        result?;

        eprintln!("DEBUG: Successfully inserted regular triple");
        Ok(())
    }

    /// Convert a StarTriple (without quoted triples) to a core RDF Triple
    pub(crate) fn convert_to_core_triple(
        &self,
        triple: &StarTriple,
    ) -> StarResult<oxirs_core::model::Triple> {
        let subject = conversion::star_term_to_subject(&triple.subject)?;
        let predicate = conversion::star_term_to_predicate(&triple.predicate)?;
        let object = conversion::star_term_to_object(&triple.object)?;

        Ok(oxirs_core::model::Triple::new(subject, predicate, object))
    }

    /// Insert a RDF-star triple (containing quoted triples) into star storage
    pub(crate) fn insert_star_triple(&self, triple: &StarTriple) -> StarResult<()> {
        let mut star_triples = self.star_triples.write().unwrap_or_else(|e| e.into_inner());
        let mut index = self
            .quoted_triple_index
            .write()
            .unwrap_or_else(|e| e.into_inner());

        let triple_index = star_triples.len();
        star_triples.push(triple.clone());

        // Build index for quoted triples
        self.index_quoted_triples(triple, triple_index, &mut index);

        debug!(
            "Inserted star triple with {} quoted triples",
            self.count_quoted_triples_in_triple(triple)
        );
        Ok(())
    }

    /// Remove a triple from the store
    pub fn remove(&self, triple: &StarTriple) -> StarResult<bool> {
        let span = span!(Level::DEBUG, "remove_triple");
        let _enter = span.enter();

        eprintln!("DEBUG: Attempting to remove triple: {triple}");
        eprintln!(
            "DEBUG: Triple contains quoted triples: {}",
            triple.contains_quoted_triples()
        );

        // First try to remove from star triples
        if triple.contains_quoted_triples() {
            let mut star_triples = self.star_triples.write().unwrap_or_else(|e| e.into_inner());

            if let Some(pos) = star_triples.iter().position(|t| t == triple) {
                star_triples.remove(pos);

                // Update all indices
                let mut index = self
                    .quoted_triple_index
                    .write()
                    .unwrap_or_else(|e| e.into_inner());

                // Update signature index
                for (_, indices) in index.signature_to_indices.iter_mut() {
                    Self::update_indices_after_removal(indices, pos);
                }

                // Update subject index
                for (_, indices) in index.subject_index.iter_mut() {
                    Self::update_indices_after_removal(indices, pos);
                }

                // Update predicate index
                for (_, indices) in index.predicate_index.iter_mut() {
                    Self::update_indices_after_removal(indices, pos);
                }

                // Update object index
                for (_, indices) in index.object_index.iter_mut() {
                    Self::update_indices_after_removal(indices, pos);
                }

                // Update nesting depth index
                for (_, indices) in index.nesting_depth_index.iter_mut() {
                    Self::update_indices_after_removal(indices, pos);
                }

                debug!("Removed star triple: {}", triple);
                return Ok(true);
            }
        } else {
            // Try to remove from core store for regular triples
            eprintln!("DEBUG: Attempting to remove regular triple from core store");
            let core_store = self.core_store.write().unwrap_or_else(|e| e.into_inner());
            if let Ok(core_triple) = self.convert_to_core_triple(triple) {
                eprintln!("DEBUG: Successfully converted to core triple");
                let core_quad = oxirs_core::model::Quad::from_triple(core_triple);
                eprintln!("DEBUG: Created core quad: {core_quad:?}");
                match CoreStore::remove_quad(&core_store, &core_quad) {
                    Ok(removed) => {
                        eprintln!("DEBUG: Core store remove_quad returned: {removed}");
                        if removed {
                            eprintln!("DEBUG: Removed regular triple: {triple}");
                            return Ok(true);
                        } else {
                            eprintln!(
                                "DEBUG: Core store remove_quad returned false - triple not found"
                            );
                        }
                    }
                    Err(e) => {
                        eprintln!("DEBUG: Core store remove_quad failed with error: {e:?}");
                    }
                }
            } else {
                eprintln!("DEBUG: Failed to convert triple to core triple");
            }
        }

        Ok(false)
    }

    /// Check if the store contains a specific triple
    pub fn contains(&self, triple: &StarTriple) -> bool {
        // First check star triples
        let star_triples = self.star_triples.read().unwrap_or_else(|e| e.into_inner());
        if star_triples.contains(triple) {
            return true;
        }

        // Then check regular triples in core store
        if !triple.contains_quoted_triples() {
            let core_store = self.core_store.read().unwrap_or_else(|e| e.into_inner());
            if let Ok(core_triple) = self.convert_to_core_triple(triple) {
                // Convert triple to quad with default graph
                let core_quad = oxirs_core::model::Quad::from_triple(core_triple);
                if let Ok(quads) = core_store.find_quads(
                    Some(core_quad.subject()),
                    Some(core_quad.predicate()),
                    Some(core_quad.object()),
                    Some(core_quad.graph_name()),
                ) {
                    return !quads.is_empty();
                }
            }
        }

        false
    }

    /// Get the number of triples in the store
    pub fn len(&self) -> usize {
        let star_triples = self.star_triples.read().unwrap_or_else(|e| e.into_inner());
        let core_store = self.core_store.read().unwrap_or_else(|e| e.into_inner());

        // Count both star triples and regular triples from core store
        let regular_count = core_store.len().unwrap_or(0);
        let star_count = star_triples.len();

        regular_count + star_count
    }

    /// Check if the store is empty
    pub fn is_empty(&self) -> bool {
        let star_triples = self.star_triples.read().unwrap_or_else(|e| e.into_inner());
        let core_store = self.core_store.read().unwrap_or_else(|e| e.into_inner());

        // Empty only if both stores are empty
        star_triples.is_empty() && core_store.is_empty().unwrap_or(true)
    }

    /// Clear all triples from the store
    pub fn clear(&self) -> StarResult<()> {
        let span = span!(Level::INFO, "clear_store");
        let _enter = span.enter();

        {
            let mut star_triples = self.star_triples.write().unwrap_or_else(|e| e.into_inner());
            star_triples.clear();
        }

        // Clear the core store by recreating it
        // Note: This is a workaround since clear_all/remove_quad have trait/impl conflicts
        {
            let mut core_store = self.core_store.write().unwrap_or_else(|e| e.into_inner());
            *core_store = CoreStore::new().map_err(StarError::CoreError)?;
        }

        {
            let mut index = self
                .quoted_triple_index
                .write()
                .unwrap_or_else(|e| e.into_inner());
            index.clear();
        }

        {
            let mut stats = self.statistics.write().unwrap_or_else(|e| e.into_inner());
            *stats = StarStatistics::default();
        }

        info!("Cleared all triples from store");
        Ok(())
    }

    /// Get statistics about the store
    pub fn statistics(&self) -> StarStatistics {
        let stats = self.statistics.read().unwrap_or_else(|e| e.into_inner());
        stats.clone()
    }

    /// Check if a triple matches the given pattern
    pub(crate) fn triple_matches(
        &self,
        triple: &StarTriple,
        subject: Option<&StarTerm>,
        predicate: Option<&StarTerm>,
        object: Option<&StarTerm>,
    ) -> bool {
        if let Some(s) = subject {
            if &triple.subject != s {
                return false;
            }
        }
        if let Some(p) = predicate {
            if &triple.predicate != p {
                return false;
            }
        }
        if let Some(o) = object {
            if &triple.object != o {
                return false;
            }
        }
        true
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
