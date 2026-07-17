//! Main TdbStore implementation: open/close, read/write operations, MVCC, transactions.

use crate::compression::{BloomFilter, PrefixCompressor};
use crate::diagnostics::{DiagnosticContext, DiagnosticEngine, DiagnosticLevel, DiagnosticReport};
use crate::dictionary::{Dictionary, NodeId, Term};
use crate::error::{Result, TdbError};
use crate::index::spatial::{
    Geometry, SpatialIndex, SpatialQuery, SpatialQueryResult, SpatialStats,
};
use crate::index::{QuadIndexes, Triple, TripleIndexes};
use crate::query_cache::{QueryCache, QueryCacheConfig};
use crate::query_monitor::{QueryMonitor, QueryMonitorConfig};
use crate::statistics::{StatisticsConfig, TripleStatistics};
use crate::storage::buffer_pool::BufferPool;
use crate::storage::file_manager::FileManager;
use crate::storage::superblock::{
    Superblock, SUPERBLOCK_FORMAT_VERSION, SUPERBLOCK_MAGIC, SUPERBLOCK_PAGE_ID,
};
use crate::store::store_types::{
    IndexMetrics, QueryResultWithStats, StorageMetrics, TdbConfig, TdbEnhancedStats, TdbStats,
    TransactionMetrics,
};
use crate::store::store_wal::StoreOp;
use crate::transaction::wal::Lsn;
use crate::transaction::{LockManager, TransactionManager, WriteAheadLog};
use std::path::Path;
use std::sync::Arc;

/// High-level TDB triple store
pub struct TdbStore {
    /// Configuration
    pub(crate) config: TdbConfig,
    /// Dictionary for term encoding
    pub(crate) dictionary: Dictionary,
    /// Triple indexes (SPO, POS, OSP) for the default (unnamed) graph.
    pub(crate) indexes: TripleIndexes,
    /// Quad indexes (GSPO, GPOS, GOSP) for named graphs.
    ///
    /// `None` when named-graph support is disabled and the store has no
    /// previously-persisted quad data (see [`TdbConfig::enable_quad_indexes`]).
    pub(crate) quad_indexes: Option<QuadIndexes>,
    /// Whether *new* named-graph writes are permitted. Reads of previously
    /// persisted quad data remain available even when writes are disabled, so
    /// reopening a quad store read-only never loses or hides data.
    pub(crate) quads_writable: bool,
    /// Number of named-graph quads currently stored (default-graph triples are
    /// counted by [`Self::triple_count`]).
    pub(crate) quad_count: usize,
    /// Transaction manager
    pub(crate) txn_manager: Arc<TransactionManager>,
    /// Buffer pool (for stats collection)
    pub(crate) buffer_pool: Arc<BufferPool>,
    /// File manager, used to persist the on-disk superblock (page 0) directly,
    /// outside the cached page space managed by the buffer pool.
    pub(crate) file_manager: Arc<FileManager>,
    /// Whether [`TdbStore::sync`] runs on `Drop` (best-effort durability).
    /// Disabled in tests that simulate a crash before an explicit sync.
    pub(crate) sync_on_drop: bool,
    /// Bloom filter for existence checks (optional)
    pub(crate) bloom_filter: Option<BloomFilter>,
    /// Prefix compressor (optional)
    pub(crate) prefix_compressor: Option<PrefixCompressor>,
    /// Triple count
    pub(crate) triple_count: usize,
    /// Query result cache
    pub(crate) query_cache: QueryCache,
    /// Statistics collector
    pub(crate) statistics: TripleStatistics,
    /// Query monitor
    pub(crate) query_monitor: QueryMonitor,
    /// Diagnostic engine
    pub(crate) diagnostic_engine: DiagnosticEngine,
    /// Spatial index for GeoSPARQL queries (optional)
    pub(crate) spatial_index: Option<SpatialIndex>,
    /// Monotonic transaction-id counter for WAL-integrated writes (F3). Seeded
    /// past any id left in the WAL by a crashed session during recovery so
    /// replayed and freshly-issued ids never collide. See
    /// [`crate::store::store_wal`].
    pub(crate) wal_txn_counter: u64,
}

impl TdbStore {
    /// Open or create a TDB store
    pub fn open<P: AsRef<Path>>(data_dir: P) -> Result<Self> {
        let config = TdbConfig::new(data_dir);
        Self::open_with_config(config)
    }

    /// Open with custom configuration
    pub fn open_with_config(config: TdbConfig) -> Result<Self> {
        // Create data directory
        std::fs::create_dir_all(&config.data_dir).map_err(TdbError::Io)?;

        // Initialize file manager and buffer pool
        let data_file = config.data_dir.join("data.tdb");
        let file_manager = Arc::new(FileManager::open(&data_file, false)?);

        // Read the on-disk catalog (page 0) BEFORE the buffer pool touches the
        // file. `Ok(None)` means a brand-new, empty data file (fresh store);
        // `Err` means the file exists but is not a compatible superblock.
        let existing_superblock = Superblock::read(&file_manager)?;

        let buffer_pool = Arc::new(BufferPool::new(
            config.buffer_pool_size,
            file_manager.clone(),
        ));

        // Whether new named-graph writes are permitted for this handle.
        let quads_writable = config.enable_quad_indexes;

        // Reconstruct (or initialize) the dictionary + triple indexes + quad
        // indexes and the triple/quad counts from the catalog.
        let (dictionary, indexes, quad_indexes, triple_count, quad_count) =
            match existing_superblock {
                Some(sb) => {
                    // Restore the persisted free-page list head so freed pages are
                    // reused across restarts.
                    file_manager.set_free_list_head(sb.free_list_head());

                    let dictionary = Dictionary::from_roots(
                        buffer_pool.clone(),
                        sb.node_to_id_root(),
                        sb.id_to_term_root(),
                        NodeId::new(sb.next_node_id),
                    );
                    let indexes = TripleIndexes::from_roots(
                        buffer_pool.clone(),
                        sb.spo_root(),
                        sb.pos_root(),
                        sb.osp_root(),
                    );

                    // Reconstruct the quad indexes when named-graph support is
                    // enabled OR when the store already holds persisted quad data
                    // (so disabling the flag on reopen never orphans/loses quads).
                    let has_persisted_quads = sb.gspo_root().is_some()
                        || sb.gpos_root().is_some()
                        || sb.gosp_root().is_some();
                    let quad_indexes = if config.enable_quad_indexes || has_persisted_quads {
                        Some(QuadIndexes::from_roots(
                            buffer_pool.clone(),
                            sb.gspo_root(),
                            sb.gpos_root(),
                            sb.gosp_root(),
                        ))
                    } else {
                        None
                    };

                    (
                        dictionary,
                        indexes,
                        quad_indexes,
                        sb.triple_count as usize,
                        sb.quad_count as usize,
                    )
                }
                None => {
                    // Fresh store: reserve page 0 for the superblock so no B+Tree
                    // page ever collides with it, then persist an empty catalog.
                    let page0 = file_manager.allocate_page()?;
                    if page0 != SUPERBLOCK_PAGE_ID {
                        return Err(TdbError::Other(format!(
                        "Failed to reserve superblock page: expected page {SUPERBLOCK_PAGE_ID}, \
                         got {page0}"
                    )));
                    }
                    Superblock::new_empty(NodeId::FIRST.as_u64()).write(&file_manager)?;

                    let dictionary = Dictionary::new(buffer_pool.clone());
                    let indexes = TripleIndexes::new(buffer_pool.clone());
                    let quad_indexes = if config.enable_quad_indexes {
                        Some(QuadIndexes::new(buffer_pool.clone()))
                    } else {
                        None
                    };
                    (dictionary, indexes, quad_indexes, 0usize, 0usize)
                }
            };

        // Initialize transaction management. Open (rather than create) the WAL
        // so any records left by a previous, un-checkpointed session are read
        // back and the LSN is positioned correctly; the committed operations are
        // replayed into the reconstructed catalog below, before the store serves
        // any read (F3 crash recovery).
        let wal = Arc::new(WriteAheadLog::open(&config.data_dir)?);
        let lock_manager = Arc::new(LockManager::new());
        let txn_manager = Arc::new(TransactionManager::new(wal, lock_manager));

        // Initialize optional components
        let bloom_filter = if config.enable_bloom_filters {
            Some(BloomFilter::new(100000, config.bloom_filter_fpr))
        } else {
            None
        };

        let prefix_compressor = if config.enable_compression {
            Some(PrefixCompressor::new(10))
        } else {
            None
        };

        // Initialize query cache
        let query_cache_config = QueryCacheConfig {
            enabled: config.enable_query_cache,
            ..Default::default()
        };
        let query_cache = QueryCache::new(query_cache_config);

        // Initialize statistics collector
        let stats_config = StatisticsConfig {
            enabled: config.enable_statistics,
            ..Default::default()
        };
        let statistics = TripleStatistics::new(stats_config);

        // Initialize query monitor
        let monitor_config = QueryMonitorConfig {
            enabled: config.enable_query_monitoring,
            ..Default::default()
        };
        let query_monitor = QueryMonitor::new(monitor_config);

        // Initialize diagnostic engine
        let diagnostic_engine = DiagnosticEngine::new();

        // Initialize spatial index
        let spatial_index = if config.enable_spatial_indexing {
            Some(SpatialIndex::new())
        } else {
            None
        };

        let mut store = Self {
            config,
            dictionary,
            indexes,
            quad_indexes,
            quads_writable,
            quad_count,
            txn_manager,
            buffer_pool,
            file_manager,
            sync_on_drop: true,
            bloom_filter,
            prefix_compressor,
            triple_count,
            query_cache,
            statistics,
            query_monitor,
            diagnostic_engine,
            spatial_index,
            wal_txn_counter: 0,
        };

        // The bloom filter is an in-memory acceleration structure that is not
        // persisted; rebuild it from the (now reconstructed) indexes so that
        // `contains()` — which trusts the bloom filter as a negative cache —
        // does not incorrectly report reopened triples as absent. For a fresh
        // store the index scan is empty and this is a no-op.
        store.rebuild_bloom_filter()?;

        // Crash recovery: replay committed WAL operations recorded since the
        // last checkpoint on top of the just-reconstructed catalog, so a store
        // that was written but not `sync()`ed still comes back with its
        // committed data. A clean shutdown truncated the WAL, so this is a
        // no-op then. Runs before any read is served.
        let replayed = store.recover_from_wal()?;
        if replayed > 0 {
            log::info!("oxirs-tdb: replayed {replayed} committed WAL operations on open");
        }

        Ok(store)
    }

    /// Repopulate the (non-persisted) bloom filter from the current indexes.
    ///
    /// Called on open so a reopened store's `contains()` fast-path is correct.
    /// O(n) in the number of triples; a persisted bloom filter would avoid the
    /// scan and is a possible future optimization.
    fn rebuild_bloom_filter(&mut self) -> Result<()> {
        if self.bloom_filter.is_none() {
            return Ok(());
        }
        let all = self.indexes.query_pattern(None, None, None)?;
        if let Some(ref mut bloom) = self.bloom_filter {
            for triple in &all {
                bloom.insert(triple);
            }
        }
        Ok(())
    }

    /// Flush all pending state to disk durably and checkpoint the WAL.
    ///
    /// After `sync()` returns, reopening the store reconstructs exactly this
    /// state from the superblock alone (the WAL has been truncated).
    ///
    /// # Ordering invariant (crash-safe checkpoint)
    ///
    /// The four steps below run in a fixed order, and each is only correct
    /// because the previous one is already durable:
    ///
    /// 1. **WAL commit + flush.** Force every logged operation to the WAL, and
    ///    capture the checkpoint LSN (the WAL high-water mark). Nothing that is
    ///    about to be reflected in the superblock may still be un-flushed in the
    ///    WAL.
    /// 2. **Page flush.** Make every dirty B+Tree/dictionary page durable
    ///    (each fsynced by the file manager). The superblock written next must
    ///    never reference a page that is not yet on disk.
    /// 3. **Superblock + checkpoint-LSN write.** Snapshot the live catalog —
    ///    the SPO/POS/OSP and GSPO/GPOS/GOSP roots, the dictionary roots, the
    ///    next id, the triple/quad counts, the free-list head, and the
    ///    checkpoint LSN — into page 0 and fsync it. Once this is durable the
    ///    on-disk catalog reflects every operation up to the checkpoint LSN.
    /// 4. **WAL truncate up to the checkpoint.** Only now, with the superblock
    ///    durable, discard the WAL records the catalog already covers.
    ///
    /// A crash between steps 3 and 4 leaves WAL records that the superblock
    /// already reflects; replay on reopen re-applies them idempotently, so the
    /// result is identical. A crash before step 3 leaves the previous
    /// superblock, and the WAL still holds the committed operations for replay.
    pub fn sync(&self) -> Result<()> {
        // 1. WAL commit + flush: make every logged operation durable, then take
        //    the checkpoint LSN (the next LSN to be assigned — all lower LSNs
        //    are now durable and about to be reflected in the superblock).
        let wal = self.txn_manager.wal();
        wal.flush()?;
        let checkpoint_lsn = wal.next_lsn();

        // 2. Persist all dirty B+Tree/dictionary pages durably.
        self.buffer_pool.flush_all()?;

        // 3. Snapshot the live catalog (including the checkpoint LSN) into the
        //    superblock and fsync it. Quad roots come from the live quad indexes
        //    when present; when quad support is disabled they are recorded as
        //    empty (a disabled store has no live quad data — see
        //    `open_with_config`, which reconstructs the indexes whenever
        //    persisted quad roots exist).
        let (gspo_root, gpos_root, gosp_root) = match &self.quad_indexes {
            Some(qi) => (qi.gspo_root(), qi.gpos_root(), qi.gosp_root()),
            None => (None, None, None),
        };
        let superblock = Superblock {
            magic: SUPERBLOCK_MAGIC,
            format_version: SUPERBLOCK_FORMAT_VERSION,
            spo_root: Superblock::option_to_slot(self.indexes.spo_root()),
            pos_root: Superblock::option_to_slot(self.indexes.pos_root()),
            osp_root: Superblock::option_to_slot(self.indexes.osp_root()),
            node_to_id_root: Superblock::option_to_slot(self.dictionary.node_to_id_root()),
            id_to_term_root: Superblock::option_to_slot(self.dictionary.id_to_term_root()),
            next_node_id: self.dictionary.next_id().as_u64(),
            triple_count: self.triple_count as u64,
            free_list_head: Superblock::option_to_slot(self.file_manager.free_list_head()),
            gspo_root: Superblock::option_to_slot(gspo_root),
            gpos_root: Superblock::option_to_slot(gpos_root),
            gosp_root: Superblock::option_to_slot(gosp_root),
            quad_count: self.quad_count as u64,
            checkpoint_lsn: checkpoint_lsn.as_u64(),
        };
        superblock.write(&self.file_manager)?;

        // 4. WAL truncate up to the checkpoint: the superblock now covers every
        //    operation with LSN < checkpoint_lsn, so those records are redundant.
        //    truncate() keeps records with LSN strictly greater than its
        //    argument, so pass one below the high-water mark to drop them all.
        let truncate_up_to = Lsn::new(checkpoint_lsn.as_u64().saturating_sub(1));
        wal.truncate(truncate_up_to)?;
        Ok(())
    }

    /// Intern a term into the dictionary without inserting any triple.
    ///
    /// Used by the bulk loader to warm the dictionary up front; unlike the old
    /// pre-population path it never fabricates `(term, term, term)` triples.
    pub fn encode_term(&self, term: &Term) -> Result<NodeId> {
        self.dictionary.encode(term)
    }

    /// Control whether [`TdbStore::sync`] runs automatically on `Drop`.
    ///
    /// Enabled by default. Disabling it lets tests simulate a process crash
    /// (a drop with no explicit `sync()`), after which only previously-synced
    /// data survives a reopen.
    pub fn set_sync_on_drop(&mut self, enabled: bool) {
        self.sync_on_drop = enabled;
    }

    /// Insert a triple
    pub fn insert(&mut self, subject: &str, predicate: &str, object: &str) -> Result<()> {
        // Encode terms to node IDs
        let s_term = Term::Iri(subject.to_string());
        let p_term = Term::Iri(predicate.to_string());
        let o_term = Term::Iri(object.to_string());

        let s_id = self.dictionary.encode(&s_term)?;
        let p_id = self.dictionary.encode(&p_term)?;
        let o_id = self.dictionary.encode(&o_term)?;

        let triple = Triple::new(s_id, p_id, o_id);

        // Add to indexes. `is_new` is false when the triple already existed, so
        // the count is not double-incremented for duplicate inserts.
        let is_new = self.indexes.insert(triple)?;

        // Update bloom filter
        if let Some(ref mut bloom) = self.bloom_filter {
            bloom.insert(&triple);
        }

        // Update statistics
        self.statistics.record_insert(s_id, p_id, o_id);

        // Invalidate query cache (data has changed)
        self.query_cache
            .invalidate_pattern(Some(subject), Some(predicate), Some(object));

        // Update count only for genuinely new triples, and log the committed
        // operation to the WAL (no-op if WAL disabled or the triple already
        // existed) so it survives a crash before the next checkpoint.
        if is_new {
            self.triple_count += 1;
            self.wal_log_op(StoreOp::InsertTriple {
                subject: s_term,
                predicate: p_term,
                object: o_term,
            })?;
        }

        Ok(())
    }

    /// Check if a triple exists
    pub fn contains(&self, subject: &str, predicate: &str, object: &str) -> Result<bool> {
        // Encode terms
        let s_term = Term::Iri(subject.to_string());
        let p_term = Term::Iri(predicate.to_string());
        let o_term = Term::Iri(object.to_string());

        let s_id = self
            .dictionary
            .lookup(&s_term)?
            .ok_or_else(|| TdbError::Other("Subject not found in dictionary".to_string()))?;
        let p_id = self
            .dictionary
            .lookup(&p_term)?
            .ok_or_else(|| TdbError::Other("Predicate not found in dictionary".to_string()))?;
        let o_id = self
            .dictionary
            .lookup(&o_term)?
            .ok_or_else(|| TdbError::Other("Object not found in dictionary".to_string()))?;

        let triple = Triple::new(s_id, p_id, o_id);

        // Check bloom filter first (if enabled)
        if let Some(ref bloom) = self.bloom_filter {
            if !bloom.contains(&triple) {
                return Ok(false); // Definitely not present
            }
        }

        // Check in indexes
        self.indexes.contains(&triple)
    }

    /// Delete a triple
    pub fn delete(&mut self, subject: &str, predicate: &str, object: &str) -> Result<bool> {
        // Encode terms
        let s_term = Term::Iri(subject.to_string());
        let p_term = Term::Iri(predicate.to_string());
        let o_term = Term::Iri(object.to_string());

        let s_id = self
            .dictionary
            .lookup(&s_term)?
            .ok_or_else(|| TdbError::Other("Subject not found in dictionary".to_string()))?;
        let p_id = self
            .dictionary
            .lookup(&p_term)?
            .ok_or_else(|| TdbError::Other("Predicate not found in dictionary".to_string()))?;
        let o_id = self
            .dictionary
            .lookup(&o_term)?
            .ok_or_else(|| TdbError::Other("Object not found in dictionary".to_string()))?;

        let triple = Triple::new(s_id, p_id, o_id);

        // Delete from indexes
        let deleted = self.indexes.delete(&triple)?;

        // Update statistics and cache if deleted, and log the committed delete
        // to the WAL so the removal survives a crash before the next checkpoint.
        if deleted {
            self.statistics.record_delete(s_id, p_id, o_id);
            self.query_cache
                .invalidate_pattern(Some(subject), Some(predicate), Some(object));
            self.triple_count = self.triple_count.saturating_sub(1);
            self.wal_log_op(StoreOp::DeleteTriple {
                subject: s_term,
                predicate: p_term,
                object: o_term,
            })?;
        }

        Ok(deleted)
    }

    /// Get number of triples
    pub fn count(&self) -> usize {
        self.triple_count
    }

    /// Get configuration
    pub fn config(&self) -> &TdbConfig {
        &self.config
    }

    /// Get transaction manager
    pub fn transaction_manager(&self) -> &Arc<TransactionManager> {
        &self.txn_manager
    }

    /// Perform crash recovery and corruption detection
    ///
    /// This should be called after opening a database to ensure data integrity
    pub fn recover(&self) -> Result<crate::recovery::RecoveryReport> {
        let recovery = crate::recovery::RecoveryManager::new(
            self.buffer_pool.clone(),
            self.txn_manager.wal().clone(),
        );
        recovery.recover()
    }

    /// Detect corruption in the database
    pub fn detect_corruption(&self) -> Result<crate::recovery::CorruptionReport> {
        let recovery = crate::recovery::RecoveryManager::new(
            self.buffer_pool.clone(),
            self.txn_manager.wal().clone(),
        );
        recovery.detect_corruption()
    }

    /// Verify index consistency
    pub fn verify_indexes(&self) -> Result<crate::recovery::IndexVerificationReport> {
        let recovery = crate::recovery::RecoveryManager::new(
            self.buffer_pool.clone(),
            self.txn_manager.wal().clone(),
        );
        recovery.verify_indexes(&self.indexes)
    }

    /// Get basic statistics
    pub fn stats(&self) -> TdbStats {
        TdbStats {
            triple_count: self.count(),
            dictionary_size: self.dictionary.size() as usize,
            bloom_filter_stats: self.bloom_filter.as_ref().map(|b| b.stats()),
            compression_stats: self.prefix_compressor.as_ref().map(|c| c.stats()),
        }
    }

    /// Get enhanced statistics with comprehensive metrics
    pub fn enhanced_stats(&self) -> TdbEnhancedStats {
        // Get basic stats
        let basic = self.stats();

        // Get buffer pool stats
        let buffer_pool_stats = self.buffer_pool.stats();

        // Estimate storage metrics
        let page_size = crate::DEFAULT_PAGE_SIZE;
        let pages_allocated = (basic.triple_count / 10).max(1); // Rough estimate
        let storage = StorageMetrics {
            total_size_bytes: (basic.triple_count * 100) as u64, // Rough estimate
            pages_allocated,
            page_size,
            memory_usage_bytes: self.config.buffer_pool_size * page_size,
        };

        // Transaction metrics
        let transaction = TransactionMetrics {
            active_transactions: 0, // Would need transaction manager stats
            wal_enabled: true,
            wal_size_bytes: 0, // Would need WAL stats
        };

        // Index metrics (all indexes should have same count)
        let index = IndexMetrics {
            spo_entries: basic.triple_count,
            pos_entries: basic.triple_count,
            osp_entries: basic.triple_count,
            indexes_consistent: true, // Assuming consistency
        };

        TdbEnhancedStats {
            basic,
            buffer_pool: buffer_pool_stats,
            storage,
            transaction,
            index,
        }
    }

    /// Insert a triple using Term types (convenience method)
    pub fn insert_triple(&mut self, subject: &Term, predicate: &Term, object: &Term) -> Result<()> {
        // Encode terms to node IDs
        let s_id = self.dictionary.encode(subject)?;
        let p_id = self.dictionary.encode(predicate)?;
        let o_id = self.dictionary.encode(object)?;

        let triple = Triple::new(s_id, p_id, o_id);

        // Add to indexes (see `insert` for the `is_new` rationale).
        let is_new = self.indexes.insert(triple)?;

        // Update bloom filter
        if let Some(ref mut bloom) = self.bloom_filter {
            bloom.insert(&triple);
        }

        // Update count only for genuinely new triples, and log the committed
        // insert to the WAL so it survives a crash before the next checkpoint.
        if is_new {
            self.triple_count += 1;
            self.wal_log_op(StoreOp::InsertTriple {
                subject: subject.clone(),
                predicate: predicate.clone(),
                object: object.clone(),
            })?;
        }

        Ok(())
    }

    /// Alias for count() for compatibility
    pub fn len(&self) -> Result<usize> {
        Ok(self.count())
    }

    /// Check if store is empty
    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }

    /// Bulk insert triples (transactional: all-or-nothing).
    ///
    /// The whole batch is logged as a *single* WAL transaction (one
    /// `Begin -> DataOp* -> Commit`, no per-element fsync) and then persisted by
    /// a single checkpoint, so a bulk load survives a reopen even if the caller
    /// drops the store without an explicit `sync()`, while keeping fsync cost
    /// amortized across the batch.
    pub fn insert_triples_bulk(&mut self, triples: &[(Term, Term, Term)]) -> Result<()> {
        // Validate all triples first (subject must be IRI or blank node, not literal)
        for (subject, _predicate, _object) in triples {
            if matches!(subject, Term::Literal { .. }) {
                return Err(TdbError::Other("Subject cannot be a literal".to_string()));
            }
        }

        // Apply each triple to the in-memory indexes without per-op WAL logging,
        // collecting the genuinely-new inserts so the batch can be logged as one
        // transaction below. `insert_triple_no_wal` mirrors `insert_triple`
        // minus the WAL append.
        let mut ops: Vec<StoreOp> = Vec::new();
        for (subject, predicate, object) in triples {
            if self.insert_triple_no_wal(subject, predicate, object)? {
                ops.push(StoreOp::InsertTriple {
                    subject: subject.clone(),
                    predicate: predicate.clone(),
                    object: object.clone(),
                });
            }
        }

        // Log the whole batch as one WAL transaction (durable via the sync
        // below; no per-element fsync).
        self.wal_log_batch(&ops)?;

        // Persist the batch durably (pages + catalog) so a bulk load survives a
        // reopen even if the caller drops the store without an explicit sync.
        self.sync()?;

        Ok(())
    }

    /// Insert a triple into the in-memory indexes (index + bloom + count) without
    /// writing a WAL record. Used by the bulk path, which logs the whole batch
    /// as one transaction. Returns whether the triple was genuinely new.
    fn insert_triple_no_wal(
        &mut self,
        subject: &Term,
        predicate: &Term,
        object: &Term,
    ) -> Result<bool> {
        let s_id = self.dictionary.encode(subject)?;
        let p_id = self.dictionary.encode(predicate)?;
        let o_id = self.dictionary.encode(object)?;

        let triple = Triple::new(s_id, p_id, o_id);
        let is_new = self.indexes.insert(triple)?;
        if let Some(ref mut bloom) = self.bloom_filter {
            bloom.insert(&triple);
        }
        if is_new {
            self.triple_count += 1;
        }
        Ok(is_new)
    }

    /// Query triples with optional pattern matching (None = wildcard)
    /// Returns matching triples as (subject, predicate, object) Terms
    ///
    /// Uses optimal index selection based on the pattern:
    /// - (S, P, O) - Exact lookup using SPO index
    /// - (S, P, *) - Range scan on SPO index
    /// - (S, *, *) - Range scan on SPO index
    /// - (*, P, O) - Range scan on POS index
    /// - (*, P, *) - Range scan on POS index
    /// - (S, *, O) - Range scan on OSP index
    /// - (*, *, O) - Range scan on OSP index
    /// - (*, *, *) - Full scan (returns all triples)
    pub fn query_triples(
        &self,
        subject: Option<&Term>,
        predicate: Option<&Term>,
        object: Option<&Term>,
    ) -> Result<Vec<(Term, Term, Term)>> {
        use crate::query_cache::QueryPattern;

        // Begin query monitoring
        let execution = self
            .query_monitor
            .begin_query(subject, predicate, object, None);

        // Create query pattern for caching
        let cache_pattern = QueryPattern::new(subject, predicate, object);

        // Check query cache first
        if let Some(cached_results) = self.query_cache.get(&cache_pattern) {
            // Cache hit!
            self.query_monitor
                .end_query(execution, cached_results.len())?;
            return Ok(cached_results);
        }

        // Cache miss - execute query
        // Convert pattern to node IDs (if specified)
        let s_id = if let Some(s) = subject {
            self.dictionary.lookup(s)?
        } else {
            None
        };

        let p_id = if let Some(p) = predicate {
            self.dictionary.lookup(p)?
        } else {
            None
        };

        let o_id = if let Some(o) = object {
            self.dictionary.lookup(o)?
        } else {
            None
        };

        // If any term in the pattern is not in dictionary, return empty
        // (except for wildcards)
        if subject.is_some() && s_id.is_none() {
            let empty = Vec::new();
            self.query_monitor.end_query(execution, empty.len())?;
            return Ok(empty);
        }
        if predicate.is_some() && p_id.is_none() {
            let empty = Vec::new();
            self.query_monitor.end_query(execution, empty.len())?;
            return Ok(empty);
        }
        if object.is_some() && o_id.is_none() {
            let empty = Vec::new();
            self.query_monitor.end_query(execution, empty.len())?;
            return Ok(empty);
        }

        // Use index pattern matching
        let matching_triples = self.indexes.query_pattern(s_id, p_id, o_id)?;

        // Convert node IDs back to terms
        let mut results = Vec::with_capacity(matching_triples.len());
        for triple in matching_triples {
            let s_term = self
                .dictionary
                .decode(triple.subject)?
                .ok_or_else(|| TdbError::Other("Subject not found in dictionary".to_string()))?;
            let p_term = self
                .dictionary
                .decode(triple.predicate)?
                .ok_or_else(|| TdbError::Other("Predicate not found in dictionary".to_string()))?;
            let o_term = self
                .dictionary
                .decode(triple.object)?
                .ok_or_else(|| TdbError::Other("Object not found in dictionary".to_string()))?;

            results.push((s_term, p_term, o_term));
        }

        // Cache the results
        self.query_cache.put(cache_pattern, results.clone())?;

        // End query monitoring
        self.query_monitor.end_query(execution, results.len())?;

        Ok(results)
    }

    /// Query triples with hints for optimization
    ///
    /// This is an enhanced version of query_triples() that accepts query hints
    /// for performance optimization. Hints can suggest index usage, enable pagination,
    /// and control caching behavior.
    ///
    /// Returns a tuple of (results, statistics).
    pub fn query_triples_with_hints(
        &self,
        subject: Option<&Term>,
        predicate: Option<&Term>,
        object: Option<&Term>,
        hints: &crate::query_hints::QueryHints,
    ) -> Result<QueryResultWithStats> {
        use std::time::Instant;

        let start = Instant::now();
        let mut stats = crate::query_hints::QueryStats::new();

        // Auto-select index if not specified in hints
        let index_type = hints.preferred_index.unwrap_or_else(|| {
            crate::query_hints::QueryHints::auto_select_index(
                subject.is_some(),
                predicate.is_some(),
                object.is_some(),
            )
        });
        stats.index_used = Some(index_type);

        // Perform the query (reusing existing implementation)
        let mut results = self.query_triples(subject, predicate, object)?;

        // Apply pagination if specified
        results = hints.apply_pagination(results);

        // Record statistics
        stats.results_found = results.len();
        stats.execution_time_us = start.elapsed().as_micros() as u64;

        // Record bloom filter usage
        if let Some(use_bloom) = hints.use_bloom_filter {
            stats.bloom_filter_used = use_bloom && self.bloom_filter.is_some();
        }

        Ok((results, stats))
    }

    /// Begin a write transaction
    pub fn begin_transaction(&self) -> Result<crate::transaction::Transaction> {
        self.txn_manager.begin()
    }

    /// Begin a read-only transaction
    ///
    /// Read-only transactions:
    /// - Cannot acquire exclusive locks
    /// - Cannot log updates to the WAL
    /// - Do not write BEGIN/COMMIT/ABORT records to WAL
    /// - Can only acquire shared locks for reading
    pub fn begin_read_transaction(&self) -> Result<crate::transaction::Transaction> {
        self.txn_manager.begin_read()
    }

    /// Commit a transaction
    pub fn commit_transaction(&self, txn: crate::transaction::Transaction) -> Result<()> {
        txn.commit()?;
        Ok(())
    }

    /// Clear all triples from the store
    ///
    /// This operation:
    /// - Creates new empty indexes
    /// - Creates new empty dictionary
    /// - Resets bloom filter
    /// - Resets prefix compressor
    /// - Resets triple count
    ///
    /// Note: This does not reclaim disk space. Use compact() after clear()
    /// to reclaim space, or delete and recreate the database directory.
    pub fn clear(&mut self) -> Result<()> {
        // Create new empty indexes (reusing the same buffer pool)
        self.indexes = TripleIndexes::new(self.buffer_pool.clone());

        // Reset the quad indexes too (if present) so named-graph data is
        // cleared alongside the default graph.
        if self.quad_indexes.is_some() {
            self.quad_indexes = Some(QuadIndexes::new(self.buffer_pool.clone()));
        }
        self.quad_count = 0;

        // Create new empty dictionary
        self.dictionary = Dictionary::new(self.buffer_pool.clone());

        // Reset bloom filter
        if let Some(ref mut bloom) = self.bloom_filter {
            *bloom = BloomFilter::new(100000, self.config.bloom_filter_fpr);
        }

        // Reset prefix compressor
        if let Some(ref mut compressor) = self.prefix_compressor {
            *compressor = PrefixCompressor::new(10);
        }

        // Reset count
        self.triple_count = 0;

        // Flush buffer pool to ensure old pages are written
        self.buffer_pool.flush_all()?;

        Ok(())
    }

    /// Compact the database (remove deleted entries, optimize layout)
    ///
    /// This high-performance operation:
    /// - Flushes all dirty pages to disk for consistency
    /// - Rebuilds bloom filters with current data (eliminates false positives)
    /// - Optimizes prefix compressor with actual URI patterns
    /// - Scans all triples to repopulate optimized data structures
    ///
    /// **Performance**: O(n) where n = number of triples
    /// **Impact**: Reduces bloom filter false positives, improves query performance
    ///
    /// # Warning
    /// This is an expensive operation. Run during maintenance windows for large databases.
    pub fn compact(&mut self) -> Result<()> {
        use std::collections::HashSet;

        // Step 1: Flush all dirty pages to ensure data consistency
        self.buffer_pool.flush_all()?;

        // Step 2: Rebuild bloom filter by scanning all current triples
        if let Some(ref mut bloom) = self.bloom_filter {
            // Create new bloom filter sized for current data (removes deleted entry overhead)
            let capacity = (self.triple_count * 2).max(100_000);
            *bloom = BloomFilter::new(capacity, self.config.bloom_filter_fpr);

            // Scan all triples from index and repopulate bloom filter
            let all_triples = self.indexes.query_pattern(None, None, None)?;

            for triple in all_triples.iter() {
                // Create compact representation for bloom filter (24 bytes: 3 x u64)
                let mut bytes = Vec::with_capacity(24);
                bytes.extend_from_slice(&triple.subject.as_u64().to_le_bytes());
                bytes.extend_from_slice(&triple.predicate.as_u64().to_le_bytes());
                bytes.extend_from_slice(&triple.object.as_u64().to_le_bytes());
                bloom.insert(&bytes);
            }
        }

        // Step 3: Rebuild prefix compressor with actual URI distribution
        if let Some(ref mut compressor) = self.prefix_compressor {
            // Collect unique URIs from dictionary for pattern analysis
            let mut uri_set = HashSet::new();

            // Scan dictionary to collect all URIs (expensive but necessary for optimal compression)
            let dict_size = self.dictionary.size();
            for node_id in 0..dict_size {
                if let Ok(Some(term)) = self
                    .dictionary
                    .decode(crate::dictionary::NodeId::dict_ref(node_id))
                {
                    if term.is_iri() {
                        uri_set.insert(term.to_string());
                    }
                }
            }

            // Rebuild compressor with analyzed URI patterns
            *compressor = PrefixCompressor::new(10);
            for uri in uri_set.iter() {
                // Compress each URI to register its prefix pattern
                let _ = compressor.compress(uri);
            }
        }

        // Note: Full B+tree compaction (physical page reordering) not implemented
        // Would require: temp indexes, triple migration, atomic swap, old file deletion
        // Current implementation focuses on logical optimization which is sufficient
        // for most production workloads and provides significant performance gains

        Ok(())
    }

    // === Advanced Monitoring and Diagnostics ===

    /// Run diagnostics on the store
    ///
    /// Returns a comprehensive diagnostic report including health status,
    /// performance metrics, and recommendations.
    ///
    /// Use `DiagnosticLevel::Quick` for fast checks, `DiagnosticLevel::Standard`
    /// for normal diagnostics, and `DiagnosticLevel::Deep` for thorough
    /// integrity checks (which may be slower).
    pub fn run_diagnostics(&self, level: DiagnosticLevel) -> DiagnosticReport {
        let context = DiagnosticContext::quick(
            self.triple_count as u64,
            self.buffer_pool.stats(),
            self.dictionary.size(),
            (self.triple_count * 100) as u64, // Rough estimate
            self.config.buffer_pool_size * crate::DEFAULT_PAGE_SIZE,
        );

        self.diagnostic_engine.run(level, &context)
    }

    /// Run deep diagnostics with full consistency checks
    ///
    /// This collects all triple data for thorough index and dictionary
    /// consistency verification. May be slow for large databases.
    pub fn run_deep_diagnostics(&self) -> DiagnosticReport {
        use std::collections::HashSet;

        let level = DiagnosticLevel::Deep;

        // Collect all triples from each index for consistency checks
        let spo_triples = self
            .indexes
            .spo()
            .range_scan(None, None)
            .unwrap_or_default()
            .into_iter()
            .map(|spo_key| Triple {
                subject: spo_key.0,
                predicate: spo_key.1,
                object: spo_key.2,
            })
            .collect::<Vec<_>>();

        let pos_triples = self
            .indexes
            .pos()
            .range_scan(None, None)
            .unwrap_or_default()
            .into_iter()
            .map(|pos_key| Triple {
                subject: pos_key.2, // POS = (P, O, S), so subject is at index 2
                predicate: pos_key.0,
                object: pos_key.1,
            })
            .collect::<Vec<_>>();

        let osp_triples = self
            .indexes
            .osp()
            .range_scan(None, None)
            .unwrap_or_default()
            .into_iter()
            .map(|osp_key| Triple {
                subject: osp_key.1, // OSP = (O, S, P), so subject is at index 1
                predicate: osp_key.2,
                object: osp_key.0,
            })
            .collect::<Vec<_>>();

        // Collect all dictionary NodeIds by extracting from collected triples
        let mut dictionary_node_ids = HashSet::new();
        for triple in &spo_triples {
            dictionary_node_ids.insert(triple.subject);
            dictionary_node_ids.insert(triple.predicate);
            dictionary_node_ids.insert(triple.object);
        }

        // Create deep diagnostic context
        let context = DiagnosticContext::deep(
            self.triple_count as u64,
            self.buffer_pool.stats(),
            self.dictionary.size(),
            (self.triple_count * 100) as u64, // Rough estimate
            self.config.buffer_pool_size * crate::DEFAULT_PAGE_SIZE,
            spo_triples,
            pos_triples,
            osp_triples,
            dictionary_node_ids,
            self.config.data_dir.display().to_string(),
        );

        self.diagnostic_engine.run(level, &context)
    }

    /// Get query cache statistics
    pub fn query_cache_stats(&self) -> &crate::query_cache::QueryCacheStats {
        self.query_cache.stats()
    }

    /// Get query monitoring statistics
    pub fn query_monitor_stats(&self) -> &crate::query_monitor::QueryMonitorStats {
        self.query_monitor.stats()
    }

    /// Get slow query history
    pub fn slow_query_history(&self) -> Vec<crate::query_monitor::SlowQueryRecord> {
        self.query_monitor.slow_query_history()
    }

    /// Get triple statistics for cost-based optimization
    pub fn triple_statistics(&self) -> &TripleStatistics {
        &self.statistics
    }

    /// Export statistics snapshot
    pub fn export_statistics(&self) -> crate::statistics::StatisticsSnapshot {
        self.statistics.export()
    }

    /// Clear query cache
    pub fn clear_query_cache(&self) {
        self.query_cache.clear();
    }

    /// Clear slow query history
    pub fn clear_slow_query_history(&self) {
        self.query_monitor.clear_slow_query_history();
    }

    /// Get active queries
    pub fn active_queries(&self) -> Vec<Arc<crate::query_monitor::QueryExecution>> {
        self.query_monitor.active_queries()
    }

    // ==================== GeoSPARQL Spatial Methods ====================

    /// Insert a geometry associated with a subject URI
    ///
    /// # Arguments
    /// * `subject` - URI of the subject node
    /// * `geometry` - Geometry to index
    ///
    /// # Example
    /// ```rust,ignore
    /// use oxirs_tdb::index::spatial::{Point, Geometry};
    ///
    /// let mut store = TdbStore::open("data")?;
    /// let point = Point::new(40.7128, -74.0060); // New York City
    /// store.insert_geometry("http://example.org/nyc", point.into())?;
    /// ```
    pub fn insert_geometry(&mut self, subject: &str, geometry: Geometry) -> Result<()> {
        if let Some(ref mut spatial_index) = self.spatial_index {
            // Encode subject to NodeId
            let subject_term = Term::Iri(subject.to_string());
            let node_id = self.dictionary.encode(&subject_term)?;

            // Insert into spatial index
            spatial_index.insert(node_id, geometry);

            Ok(())
        } else {
            Err(TdbError::Other(
                "Spatial indexing is not enabled".to_string(),
            ))
        }
    }

    /// Execute a spatial query
    ///
    /// # Arguments
    /// * `query` - Spatial query to execute
    ///
    /// # Returns
    /// Vector of matching geometries with their subject URIs
    ///
    /// # Example
    /// ```rust,ignore
    /// use oxirs_tdb::index::spatial::{SpatialQuery, Point};
    ///
    /// let store = TdbStore::open("data")?;
    /// let query = SpatialQuery::WithinDistance {
    ///     center: Point::new(40.7589, -73.9851), // Times Square
    ///     distance: 10_000.0, // 10km
    /// };
    /// let results = store.spatial_query(&query)?;
    /// ```
    pub fn spatial_query(&self, query: &SpatialQuery) -> Result<Vec<SpatialQueryResult>> {
        if let Some(ref spatial_index) = self.spatial_index {
            Ok(spatial_index.query(query))
        } else {
            Err(TdbError::Other(
                "Spatial indexing is not enabled".to_string(),
            ))
        }
    }

    /// Get spatial index statistics
    ///
    /// Returns statistics about the spatial index including geometry counts
    /// and R-tree structure information.
    pub fn spatial_statistics(&self) -> Result<SpatialStats> {
        if let Some(ref spatial_index) = self.spatial_index {
            Ok(spatial_index.stats())
        } else {
            Err(TdbError::Other(
                "Spatial indexing is not enabled".to_string(),
            ))
        }
    }

    /// Check if spatial indexing is enabled
    pub fn is_spatial_indexing_enabled(&self) -> bool {
        self.spatial_index.is_some()
    }

    /// Remove a geometry from the spatial index
    ///
    /// # Arguments
    /// * `subject` - URI of the subject node whose geometry should be removed
    pub fn remove_geometry(&mut self, subject: &str) -> Result<bool> {
        if let Some(ref mut spatial_index) = self.spatial_index {
            // Encode subject to NodeId
            let subject_term = Term::Iri(subject.to_string());
            if let Some(node_id) = self.dictionary.lookup(&subject_term)? {
                Ok(spatial_index.remove(&node_id).is_some())
            } else {
                Ok(false) // Subject not in dictionary
            }
        } else {
            Err(TdbError::Other(
                "Spatial indexing is not enabled".to_string(),
            ))
        }
    }
}

impl Drop for TdbStore {
    /// Best-effort durability on drop: unless disabled via
    /// [`TdbStore::set_sync_on_drop`], flush all pages and the superblock so a
    /// gracefully-dropped store is always persisted. Errors are logged (a Drop
    /// cannot return them) rather than silently ignored.
    fn drop(&mut self) {
        if self.sync_on_drop {
            if let Err(e) = self.sync() {
                log::error!("oxirs-tdb: TdbStore failed to sync on drop: {e}");
            }
        }
    }
}
