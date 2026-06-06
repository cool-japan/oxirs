//! Indexing, bulk insertion, optimization, and supporting infrastructure.
//!
//! This sibling module of `crate::store` contains the maintenance side of
//! [`StarStore`]: index construction, bulk-insert pipelines, cache integration,
//! detailed statistics, memory-mapped storage hooks, and the connection-pool
//! factory.

use std::collections::BTreeSet;
use std::thread;
use std::time::Instant;

use tracing::{debug, info, span, Level};

use crate::model::{StarTerm, StarTriple};
use crate::store::index::{IndexStatistics, QuotedTripleIndex};
use crate::store::{bulk_insert_mod, cache_mod, pool_mod, StarStore};
use crate::{StarError, StarResult, StarStatistics};

use bulk_insert_mod::BulkInsertConfig;
use cache_mod::CacheStatistics;
use pool_mod::ConnectionPool;

impl StarStore {
    /// Build index entries for quoted triples in a given triple using B-tree indices
    pub(crate) fn index_quoted_triples(
        &self,
        triple: &StarTriple,
        triple_index: usize,
        index: &mut QuotedTripleIndex,
    ) {
        self.index_quoted_triples_recursive(triple, triple_index, index);

        // Index by nesting depth for performance optimization
        let depth = triple.nesting_depth();
        index
            .nesting_depth_index
            .entry(depth)
            .or_default()
            .insert(triple_index);
    }

    /// Recursively index quoted triples with multi-dimensional indexing
    pub(crate) fn index_quoted_triples_recursive(
        &self,
        triple: &StarTriple,
        triple_index: usize,
        index: &mut QuotedTripleIndex,
    ) {
        // Index quoted triples in subject
        if let StarTerm::QuotedTriple(qt) = &triple.subject {
            let signature = self.quoted_triple_key(qt);
            index
                .signature_to_indices
                .entry(signature)
                .or_default()
                .insert(triple_index);

            // Index by subject signature for S?? queries
            let subject_key = format!("SUBJ:{}", qt.subject);
            index
                .subject_index
                .entry(subject_key)
                .or_default()
                .insert(triple_index);

            // Recursively index nested quoted triples
            self.index_quoted_triples_recursive(qt, triple_index, index);
        }

        // Index quoted triples in predicate (rare but possible in some extensions)
        if let StarTerm::QuotedTriple(qt) = &triple.predicate {
            let signature = self.quoted_triple_key(qt);
            index
                .signature_to_indices
                .entry(signature)
                .or_default()
                .insert(triple_index);

            // Index by predicate signature for ?P? queries
            let predicate_key = format!("PRED:{}", qt.predicate);
            index
                .predicate_index
                .entry(predicate_key)
                .or_default()
                .insert(triple_index);

            // ALSO index the subject and object of the quoted triple found in predicate position
            let qt_subject_key = format!("SUBJ:{}", qt.subject);
            index
                .subject_index
                .entry(qt_subject_key)
                .or_default()
                .insert(triple_index);

            let qt_object_key = format!("OBJ:{}", qt.object);
            index
                .object_index
                .entry(qt_object_key)
                .or_default()
                .insert(triple_index);

            // Recursively index nested quoted triples
            self.index_quoted_triples_recursive(qt, triple_index, index);
        }

        // Index quoted triples in object
        if let StarTerm::QuotedTriple(qt) = &triple.object {
            let signature = self.quoted_triple_key(qt);
            index
                .signature_to_indices
                .entry(signature)
                .or_default()
                .insert(triple_index);

            // Index by object signature for ??O queries
            let object_key = format!("OBJ:{}", qt.object);
            index
                .object_index
                .entry(object_key)
                .or_default()
                .insert(triple_index);

            // ALSO index the subject and predicate of the quoted triple found in object position
            // This allows finding triples like "bob believes <<alice age 25>>" when searching for alice
            let qt_subject_key = format!("SUBJ:{}", qt.subject);
            index
                .subject_index
                .entry(qt_subject_key)
                .or_default()
                .insert(triple_index);

            let qt_predicate_key = format!("PRED:{}", qt.predicate);
            index
                .predicate_index
                .entry(qt_predicate_key)
                .or_default()
                .insert(triple_index);

            // Recursively index nested quoted triples
            self.index_quoted_triples_recursive(qt, triple_index, index);
        }
    }

    /// Generate a key for indexing quoted triples
    pub(crate) fn quoted_triple_key(&self, triple: &StarTriple) -> String {
        format!("{}|{}|{}", triple.subject, triple.predicate, triple.object)
    }

    /// Update indices after removing an item at position `pos`
    /// This efficiently updates all indices > pos by decrementing them
    pub(crate) fn update_indices_after_removal(indices: &mut BTreeSet<usize>, pos: usize) {
        // Remove the item at pos
        indices.remove(&pos);

        // Create a new set with updated indices
        let updated: BTreeSet<usize> = indices
            .iter()
            .map(|&idx| if idx > pos { idx - 1 } else { idx })
            .collect();

        // Replace the old set with the updated one
        *indices = updated;
    }

    /// Count quoted triples within a single triple
    #[allow(clippy::only_used_in_recursion)]
    pub(crate) fn count_quoted_triples_in_triple(&self, triple: &StarTriple) -> usize {
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

    /// Optimize the store by rebuilding indices
    pub fn optimize(&self) -> StarResult<()> {
        let span = span!(Level::INFO, "optimize_store");
        let _enter = span.enter();

        let star_triples = self.star_triples.read().unwrap_or_else(|e| e.into_inner());
        let mut index = self
            .quoted_triple_index
            .write()
            .unwrap_or_else(|e| e.into_inner());

        // Rebuild the quoted triple index with all new B-tree structures
        index.clear();
        for (i, triple) in star_triples.iter().enumerate() {
            if triple.contains_quoted_triples() {
                self.index_quoted_triples(triple, i, &mut index);
            }
        }

        // Compact the indices by removing empty entries
        index
            .signature_to_indices
            .retain(|_, indices| !indices.is_empty());
        index.subject_index.retain(|_, indices| !indices.is_empty());
        index
            .predicate_index
            .retain(|_, indices| !indices.is_empty());
        index.object_index.retain(|_, indices| !indices.is_empty());
        index
            .nesting_depth_index
            .retain(|_, indices| !indices.is_empty());

        info!(
            "Store optimization completed - rebuilt {} index entries",
            index.signature_to_indices.len()
                + index.subject_index.len()
                + index.predicate_index.len()
                + index.object_index.len()
                + index.nesting_depth_index.len()
        );
        Ok(())
    }

    /// Bulk insert triples with optimized performance
    pub fn bulk_insert(&self, triples: &[StarTriple], config: &BulkInsertConfig) -> StarResult<()> {
        let span = span!(Level::INFO, "bulk_insert", count = triples.len());
        let _enter = span.enter();

        info!("Starting bulk insertion of {} triples", triples.len());
        let start_time = Instant::now();

        // Enable bulk mode
        {
            let mut bulk_state = self
                .bulk_insert_state
                .write()
                .unwrap_or_else(|e| e.into_inner());
            bulk_state.active = true;
            bulk_state.pending_triples.clear();
            bulk_state.current_memory_usage = 0;
            bulk_state.batch_count = 0;
        }

        if config.parallel_processing && triples.len() > config.batch_size {
            self.bulk_insert_parallel(triples, config)?;
        } else {
            self.bulk_insert_sequential(triples, config)?;
        }

        // Finalize bulk insertion
        self.finalize_bulk_insert(config)?;

        let elapsed = start_time.elapsed();
        info!(
            "Bulk insertion completed in {:?} for {} triples",
            elapsed,
            triples.len()
        );

        // Update statistics
        {
            let mut stats = self.statistics.write().unwrap_or_else(|e| e.into_inner());
            stats.processing_time_us += elapsed.as_micros() as u64;
        }

        Ok(())
    }

    /// Sequential bulk insertion implementation
    pub(crate) fn bulk_insert_sequential(
        &self,
        triples: &[StarTriple],
        config: &BulkInsertConfig,
    ) -> StarResult<()> {
        for batch in triples.chunks(config.batch_size) {
            for triple in batch {
                // Validate the triple
                triple.validate()?;

                // Insert based on triple type
                if triple.contains_quoted_triples() {
                    if config.defer_index_updates {
                        // Add to pending list for later indexing
                        let mut bulk_state = self
                            .bulk_insert_state
                            .write()
                            .unwrap_or_else(|e| e.into_inner());
                        bulk_state.pending_triples.push(triple.clone());
                        bulk_state.current_memory_usage += self.estimate_triple_memory_size(triple);
                    } else {
                        self.insert_star_triple(triple)?;
                    }
                } else {
                    self.insert_regular_triple(triple)?;
                }
            }

            // Check memory threshold
            {
                let bulk_state = self
                    .bulk_insert_state
                    .read()
                    .unwrap_or_else(|e| e.into_inner());
                if bulk_state.current_memory_usage >= config.memory_threshold {
                    drop(bulk_state);
                    self.flush_pending_triples(config)?;
                }
            }

            // Update batch count
            {
                let mut bulk_state = self
                    .bulk_insert_state
                    .write()
                    .unwrap_or_else(|e| e.into_inner());
                bulk_state.batch_count += 1;
            }
        }

        Ok(())
    }

    /// Parallel bulk insertion implementation
    pub(crate) fn bulk_insert_parallel(
        &self,
        triples: &[StarTriple],
        config: &BulkInsertConfig,
    ) -> StarResult<()> {
        let chunk_size = triples.len() / config.worker_threads;
        let mut handles = Vec::new();

        for chunk in triples.chunks(chunk_size) {
            let chunk = chunk.to_vec();
            let store_clone = self.clone();
            let config_clone = config.clone();

            let handle =
                thread::spawn(move || store_clone.bulk_insert_sequential(&chunk, &config_clone));
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle
                .join()
                .map_err(|e| StarError::query_error(format!("Thread join error: {e:?}")))??;
        }

        Ok(())
    }

    /// Flush pending triples and rebuild indices
    pub(crate) fn flush_pending_triples(&self, config: &BulkInsertConfig) -> StarResult<()> {
        let pending_triples = {
            let mut bulk_state = self
                .bulk_insert_state
                .write()
                .unwrap_or_else(|e| e.into_inner());
            let triples = bulk_state.pending_triples.clone();
            bulk_state.pending_triples.clear();
            bulk_state.current_memory_usage = 0;
            triples
        };

        if !pending_triples.is_empty() {
            debug!("Flushing {} pending triples", pending_triples.len());

            // Insert all pending triples into storage
            {
                let mut star_triples = self.star_triples.write().unwrap_or_else(|e| e.into_inner());
                let base_index = star_triples.len();
                star_triples.extend(pending_triples.clone());

                // Build indices for the new triples
                if !config.defer_index_updates {
                    let mut index = self
                        .quoted_triple_index
                        .write()
                        .unwrap_or_else(|e| e.into_inner());
                    for (i, triple) in pending_triples.iter().enumerate() {
                        self.index_quoted_triples(triple, base_index + i, &mut index);
                    }
                }
            }
        }

        Ok(())
    }

    /// Finalize bulk insertion by rebuilding indices if needed
    pub(crate) fn finalize_bulk_insert(&self, config: &BulkInsertConfig) -> StarResult<()> {
        // Flush any remaining pending triples
        self.flush_pending_triples(config)?;

        // Rebuild indices if they were deferred
        if config.defer_index_updates {
            info!("Rebuilding indices after bulk insertion");
            self.optimize()?;
        }

        // Reset bulk state
        {
            let mut bulk_state = self
                .bulk_insert_state
                .write()
                .unwrap_or_else(|e| e.into_inner());
            bulk_state.active = false;
            bulk_state.pending_triples.clear();
            bulk_state.current_memory_usage = 0;
            bulk_state.batch_count = 0;
        }

        Ok(())
    }

    /// Estimate memory size of a triple for memory tracking
    pub(crate) fn estimate_triple_memory_size(&self, triple: &StarTriple) -> usize {
        // Rough estimation based on string lengths and structure
        let subject_size = match &triple.subject {
            StarTerm::NamedNode(nn) => nn.iri.len(),
            StarTerm::BlankNode(bn) => bn.id.len(),
            StarTerm::Literal(lit) => lit.value.len(),
            StarTerm::QuotedTriple(_) => 200, // Estimated overhead
            StarTerm::Variable(var) => var.name.len(),
        };

        let predicate_size = match &triple.predicate {
            StarTerm::NamedNode(nn) => nn.iri.len(),
            _ => 50, // Default estimate
        };

        let object_size = match &triple.object {
            StarTerm::NamedNode(nn) => nn.iri.len(),
            StarTerm::BlankNode(bn) => bn.id.len(),
            StarTerm::Literal(lit) => lit.value.len(),
            StarTerm::QuotedTriple(_) => 200, // Estimated overhead
            StarTerm::Variable(var) => var.name.len(),
        };

        subject_size + predicate_size + object_size + 100 // Base overhead
    }

    /// Enable memory-mapped storage
    pub fn enable_memory_mapping(
        &self,
        file_path: &str,
        enable_compression: bool,
    ) -> StarResult<()> {
        let span = span!(Level::INFO, "enable_memory_mapping");
        let _enter = span.enter();

        info!("Enabling memory-mapped storage at: {}", file_path);

        {
            let mut mm_state = self
                .memory_mapped
                .write()
                .unwrap_or_else(|e| e.into_inner());
            mm_state.enabled = true;
            mm_state.file_path = Some(file_path.to_string());
            mm_state.compression_enabled = enable_compression;
            mm_state.last_sync = Some(Instant::now());
        }

        // In a full implementation, this would set up actual memory mapping
        // For now, we just track the state
        info!(
            "Memory-mapped storage enabled with compression: {}",
            enable_compression
        );
        Ok(())
    }

    /// Get optimized triples using cache
    pub fn get_triples_cached(&self, pattern: &str) -> Vec<StarTriple> {
        let span = span!(Level::DEBUG, "get_triples_cached");
        let _enter = span.enter();

        // Check cache first
        if let Some(cached_results) = self.cache.get(pattern) {
            debug!("Cache hit for pattern: {}", pattern);
            return cached_results;
        }

        // Cache miss - compute results
        debug!("Cache miss for pattern: {}", pattern);
        let results = self.compute_pattern_results(pattern);

        // Store in cache
        self.cache.put(pattern.to_string(), results.clone());

        results
    }

    /// Compute pattern results (placeholder implementation)
    pub(crate) fn compute_pattern_results(&self, pattern: &str) -> Vec<StarTriple> {
        // This is a simplified implementation
        // In practice, this would parse the pattern and execute the query
        if pattern.contains("quoted") {
            self.find_triples_by_nesting_depth(1, None)
        } else {
            self.triples()
        }
    }

    /// Get comprehensive storage statistics
    pub fn get_detailed_statistics(&self) -> DetailedStorageStatistics {
        let base_stats = self.statistics();
        let cache_stats = self.cache.get_statistics();
        let index_stats = {
            let index = self
                .quoted_triple_index
                .read()
                .unwrap_or_else(|e| e.into_inner());
            index.get_statistics()
        };
        let bulk_state = self
            .bulk_insert_state
            .read()
            .unwrap_or_else(|e| e.into_inner());
        let mm_state = self.memory_mapped.read().unwrap_or_else(|e| e.into_inner());

        DetailedStorageStatistics {
            basic_stats: base_stats,
            cache_stats,
            index_stats,
            bulk_insert_active: bulk_state.active,
            bulk_pending_count: bulk_state.pending_triples.len(),
            bulk_memory_usage: bulk_state.current_memory_usage,
            memory_mapped_enabled: mm_state.enabled,
            memory_mapped_path: mm_state.file_path.clone(),
        }
    }

    /// Create a connection pool for this store type
    pub fn create_connection_pool(
        max_connections: usize,
        config: crate::StarConfig,
    ) -> ConnectionPool {
        ConnectionPool::new(max_connections, config)
    }

    /// Compress stored data (placeholder implementation)
    pub fn compress_storage(&self) -> StarResult<usize> {
        let span = span!(Level::INFO, "compress_storage");
        let _enter = span.enter();

        // In a full implementation, this would compress the stored triples
        let triple_count = self.len();
        info!("Compressed storage for {} triples", triple_count);

        // Return estimated space saved (placeholder)
        Ok(triple_count * 50)
    }
}

/// Comprehensive storage statistics including optimizations
#[derive(Debug, Clone)]
pub struct DetailedStorageStatistics {
    pub basic_stats: StarStatistics,
    pub cache_stats: CacheStatistics,
    pub index_stats: IndexStatistics,
    pub bulk_insert_active: bool,
    pub bulk_pending_count: usize,
    pub bulk_memory_usage: usize,
    pub memory_mapped_enabled: bool,
    pub memory_mapped_path: Option<String>,
}
