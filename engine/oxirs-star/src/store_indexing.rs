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
        // Guard against config.worker_threads > triples.len() (or 0 worker
        // threads), which would otherwise compute a chunk_size of 0 and make
        // `.chunks(0)` panic.
        let worker_threads = config.worker_threads.max(1);
        let chunk_size = ((triples.len() + worker_threads - 1) / worker_threads).max(1);
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

    /// Enable memory-mapped storage.
    ///
    /// `StarStore` keeps its triples in an in-process `Vec<StarTriple>` plus
    /// B-tree indices; there is currently no disk-backed representation to
    /// map into memory, so this cannot honestly reduce RAM usage for large
    /// datasets. Rather than flip an `enabled` flag while silently keeping
    /// everything fully resident in memory (misleading for memory-constrained
    /// deployments), this fails loudly until real mmap-backed storage is
    /// implemented.
    ///
    /// For genuine disk-backed / memory-mapped storage today, use
    /// [`crate::storage_integration::StarStorageBackend::memory_mapped`],
    /// which constructs a separate backend with real on-disk persistence
    /// hooks, instead of retrofitting an existing in-memory `StarStore`.
    pub fn enable_memory_mapping(
        &self,
        file_path: &str,
        _enable_compression: bool,
    ) -> StarResult<()> {
        let span = span!(Level::INFO, "enable_memory_mapping");
        let _enter = span.enter();

        Err(StarError::configuration_error(format!(
            "Memory-mapped storage is not implemented for StarStore (requested path: \
             '{file_path}'): triples are always held fully in memory. Use \
             storage_integration::StarStorageBackend::memory_mapped for a disk-backed \
             backend instead of enable_memory_mapping()."
        )))
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

    /// Compute pattern results by parsing `pattern` into subject/predicate/
    /// object components and delegating to the store's (indexed) [`query`](Self::query).
    ///
    /// Supported syntax: three whitespace-separated terms
    /// `SUBJECT PREDICATE OBJECT`, where each term is either `?` (wildcard,
    /// matches anything) or an N-Triples-style term:
    /// - `<http://example.org/iri>` for an IRI
    /// - `_:label` for a blank node
    /// - `"value"`, `"value"@lang`, or `"value"^^<datatype-iri>` for a literal
    ///
    /// The bare keyword `quoted` is retained for backward compatibility and
    /// returns all triples containing at least one quoted triple. Patterns
    /// that cannot be parsed fall back to a full scan (logged at debug
    /// level) rather than silently returning the wrong result set.
    pub(crate) fn compute_pattern_results(&self, pattern: &str) -> Vec<StarTriple> {
        let trimmed = pattern.trim();

        if trimmed == "quoted" {
            return self.find_triples_by_nesting_depth(1, None);
        }

        match Self::parse_triple_pattern(trimmed) {
            Some((subject, predicate, object)) => self
                .query(subject.as_ref(), predicate.as_ref(), object.as_ref())
                .unwrap_or_default(),
            None => {
                debug!(
                    "Could not parse triple pattern '{}'; falling back to full scan",
                    pattern
                );
                self.triples()
            }
        }
    }

    /// Parse a `"SUBJECT PREDICATE OBJECT"` pattern string into optional
    /// [`StarTerm`] filters (`None` = wildcard `?`). Returns `None` if the
    /// pattern does not tokenize into exactly three well-formed terms.
    fn parse_triple_pattern(
        pattern: &str,
    ) -> Option<(Option<StarTerm>, Option<StarTerm>, Option<StarTerm>)> {
        let tokens = Self::tokenize_pattern(pattern)?;
        if tokens.len() != 3 {
            return None;
        }

        let subject = Self::parse_pattern_term(&tokens[0])?;
        let predicate = Self::parse_pattern_term(&tokens[1])?;
        let object = Self::parse_pattern_term(&tokens[2])?;
        Some((subject, predicate, object))
    }

    /// Split a pattern string into whitespace-separated terms, treating
    /// `<...>` IRIs and `"..."` literals (with optional `@lang`/`^^<dt>`
    /// suffix) as atomic tokens even if they cannot contain whitespace
    /// themselves (they never legally do for IRIs/simple literals here).
    fn tokenize_pattern(pattern: &str) -> Option<Vec<String>> {
        let chars: Vec<char> = pattern.chars().collect();
        let mut tokens = Vec::new();
        let mut i = 0;

        while i < chars.len() {
            while i < chars.len() && chars[i].is_whitespace() {
                i += 1;
            }
            if i >= chars.len() {
                break;
            }

            let start = i;
            match chars[i] {
                '<' => {
                    while i < chars.len() && chars[i] != '>' {
                        i += 1;
                    }
                    if i >= chars.len() {
                        return None; // Unterminated IRI
                    }
                    i += 1; // include '>'
                }
                '"' => {
                    i += 1;
                    while i < chars.len() && chars[i] != '"' {
                        if chars[i] == '\\' && i + 1 < chars.len() {
                            i += 1;
                        }
                        i += 1;
                    }
                    if i >= chars.len() {
                        return None; // Unterminated literal
                    }
                    i += 1; // include closing quote

                    if i < chars.len() && chars[i] == '@' {
                        i += 1;
                        while i < chars.len() && !chars[i].is_whitespace() {
                            i += 1;
                        }
                    } else if i + 1 < chars.len() && chars[i] == '^' && chars[i + 1] == '^' {
                        i += 2;
                        if i < chars.len() && chars[i] == '<' {
                            while i < chars.len() && chars[i] != '>' {
                                i += 1;
                            }
                            if i >= chars.len() {
                                return None; // Unterminated datatype IRI
                            }
                            i += 1;
                        }
                    }
                }
                _ => {
                    while i < chars.len() && !chars[i].is_whitespace() {
                        i += 1;
                    }
                }
            }

            tokens.push(chars[start..i].iter().collect());
        }

        Some(tokens)
    }

    /// Parse a single pattern token into an optional [`StarTerm`] filter.
    /// Returns `Some(None)` for the wildcard `?`, `Some(Some(term))` for a
    /// successfully parsed term, and `None` if the token is malformed.
    fn parse_pattern_term(token: &str) -> Option<Option<StarTerm>> {
        if token == "?" {
            return Some(None);
        }

        if let Some(iri) = token.strip_prefix('<').and_then(|s| s.strip_suffix('>')) {
            return StarTerm::iri(iri).ok().map(Some);
        }

        if let Some(label) = token.strip_prefix("_:") {
            return StarTerm::blank_node(label).ok().map(Some);
        }

        if token.starts_with('"') {
            return Self::parse_literal_token(token).map(Some);
        }

        None
    }

    /// Parse a literal token (`"value"`, `"value"@lang`, or
    /// `"value"^^<datatype-iri>`) into a [`StarTerm::Literal`].
    fn parse_literal_token(token: &str) -> Option<StarTerm> {
        let chars: Vec<char> = token.chars().collect();
        if chars.first() != Some(&'"') {
            return None;
        }

        let mut i = 1;
        while i < chars.len() && chars[i] != '"' {
            if chars[i] == '\\' && i + 1 < chars.len() {
                i += 1;
            }
            i += 1;
        }
        if i >= chars.len() {
            return None;
        }

        let value: String = chars[1..i].iter().collect();
        let rest: String = chars[i + 1..].iter().collect();

        if let Some(lang) = rest.strip_prefix('@') {
            return StarTerm::literal_with_language(&value, lang).ok();
        }

        if let Some(dt) = rest.strip_prefix("^^") {
            let dt_iri = dt.strip_prefix('<').and_then(|s| s.strip_suffix('>'))?;
            return StarTerm::literal_with_datatype(&value, dt_iri).ok();
        }

        if rest.is_empty() {
            return StarTerm::literal(&value).ok();
        }

        None
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

    /// Compress the store's serialized data and report the real number of
    /// bytes saved.
    ///
    /// This serializes the current triple set with `oxicode` (per the
    /// COOLJAPAN no-bincode policy) and compresses it with `oxiarc-zstd`
    /// (per the COOLJAPAN compression policy), returning the actual
    /// measured difference between the uncompressed and compressed byte
    /// lengths rather than a fabricated estimate. This does not persist the
    /// compressed bytes anywhere; it is a storage-savings measurement, not a
    /// durable compaction operation.
    pub fn compress_storage(&self) -> StarResult<usize> {
        let span = span!(Level::INFO, "compress_storage");
        let _enter = span.enter();

        let triples = self.triples();
        let serialized = oxicode::serde::encode_to_vec(&triples, oxicode::config::standard())
            .map_err(|e| {
                StarError::query_error(format!("Failed to serialize triples for compression: {e}"))
            })?;

        let compressed = oxiarc_zstd::encode_all(&serialized, 3).map_err(|e| {
            StarError::query_error(format!("Failed to compress serialized triples: {e}"))
        })?;

        let space_saved = serialized.len().saturating_sub(compressed.len());

        info!(
            "Compressed storage for {} triples: {} bytes -> {} bytes ({} bytes saved)",
            triples.len(),
            serialized.len(),
            compressed.len(),
            space_saved
        );

        Ok(space_saved)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{StarTerm, StarTriple};
    use crate::store::StarStore;
    use crate::StarResult;

    /// Regression test: bulk_insert_parallel must not panic on
    /// `.chunks(0)` when worker_threads exceeds the number of triples (or
    /// is otherwise larger than what would produce a non-zero chunk size).
    #[test]
    fn test_bulk_insert_parallel_does_not_panic_with_few_triples() -> StarResult<()> {
        let store = StarStore::new();

        let triples: Vec<StarTriple> = (0..3)
            .map(|i| {
                StarTriple::new(
                    StarTerm::iri(&format!("http://example.org/s{i}")).unwrap(),
                    StarTerm::iri("http://example.org/p").unwrap(),
                    StarTerm::literal(&format!("o{i}")).unwrap(),
                )
            })
            .collect();

        let config = BulkInsertConfig {
            batch_size: 1,
            defer_index_updates: false,
            memory_threshold: usize::MAX,
            parallel_processing: true,
            // More worker threads than triples: chunk_size used to compute
            // to 3 / 16 == 0, which made `.chunks(0)` panic.
            worker_threads: 16,
        };

        store.bulk_insert_parallel(&triples, &config)?;
        assert_eq!(store.len(), 3);

        Ok(())
    }

    #[test]
    fn test_compute_pattern_results_parses_bound_pattern() -> StarResult<()> {
        let store = StarStore::new();

        let alice = StarTerm::iri("http://example.org/alice")?;
        let knows = StarTerm::iri("http://example.org/knows")?;
        let bob = StarTerm::iri("http://example.org/bob")?;
        let matching = StarTriple::new(alice.clone(), knows.clone(), bob.clone());
        let non_matching = StarTriple::new(
            bob,
            StarTerm::iri("http://example.org/likes")?,
            alice.clone(),
        );

        store.insert(&matching)?;
        store.insert(&non_matching)?;

        // Bound subject+predicate, wildcard object: only `matching` triple.
        let pattern = "<http://example.org/alice> <http://example.org/knows> ?";
        let results = store.compute_pattern_results(pattern);
        assert_eq!(results, vec![matching]);

        Ok(())
    }

    #[test]
    fn test_compute_pattern_results_quoted_keyword_backward_compat() -> StarResult<()> {
        let store = StarStore::new();

        let inner = StarTriple::new(
            StarTerm::iri("http://example.org/s1")?,
            StarTerm::iri("http://example.org/p1")?,
            StarTerm::literal("o1")?,
        );
        let quoted_triple = StarTriple::new(
            StarTerm::quoted_triple(inner),
            StarTerm::iri("http://example.org/certainty")?,
            StarTerm::literal("0.9")?,
        );
        store.insert(&quoted_triple)?;

        let results = store.compute_pattern_results("quoted");
        assert_eq!(results, vec![quoted_triple]);

        Ok(())
    }

    #[test]
    fn test_compute_pattern_results_unparseable_falls_back_to_full_scan() -> StarResult<()> {
        let store = StarStore::new();
        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/s")?,
            StarTerm::iri("http://example.org/p")?,
            StarTerm::literal("o")?,
        );
        store.insert(&triple)?;

        // Malformed pattern (unterminated IRI): falls back to a full scan
        // instead of silently returning an empty / wrong result set.
        let results = store.compute_pattern_results("<http://example.org/s ? ?");
        assert_eq!(results, vec![triple]);

        Ok(())
    }

    #[test]
    fn test_compress_storage_reports_real_measurement() -> StarResult<()> {
        let store = StarStore::new();
        for i in 0..20 {
            store.insert(&StarTriple::new(
                StarTerm::iri(&format!("http://example.org/s{i}"))?,
                StarTerm::iri("http://example.org/p")?,
                StarTerm::literal("a moderately repetitive literal value for compression")?,
            ))?;
        }

        // Highly repetitive data should compress to a non-trivial number of
        // saved bytes (previously this was always `triple_count * 50`
        // regardless of actual content).
        let space_saved = store.compress_storage()?;
        assert!(space_saved > 0);

        Ok(())
    }

    #[test]
    fn test_enable_memory_mapping_fails_loudly() {
        let store = StarStore::new();
        let result = store.enable_memory_mapping("/tmp/does-not-matter", false);
        assert!(result.is_err());
    }
}
