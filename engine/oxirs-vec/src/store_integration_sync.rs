use crate::embeddings::EmbeddableContent;
use crate::store_integration_types::*;
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

impl TransactionManager {
    pub fn new(config: StoreIntegrationConfig) -> Self {
        Self {
            active_transactions: std::sync::Arc::new(parking_lot::RwLock::new(HashMap::new())),
            transaction_counter: AtomicU64::new(1),
            config,
            write_ahead_log: std::sync::Arc::new(WriteAheadLog::new()),
            lock_manager: std::sync::Arc::new(LockManager::new()),
        }
    }

    pub fn begin_transaction(&self, isolation_level: IsolationLevel) -> Result<TransactionId> {
        let transaction_id = self.transaction_counter.fetch_add(1, Ordering::Relaxed);
        let transaction = Transaction {
            id: transaction_id,
            start_time: std::time::SystemTime::now(),
            timeout: self.config.transaction_timeout,
            operations: Vec::new(),
            status: TransactionStatus::Active,
            isolation_level,
            read_set: std::collections::HashSet::new(),
            write_set: std::collections::HashSet::new(),
        };

        let mut active_txns = self.active_transactions.write();
        active_txns.insert(transaction_id, transaction);

        Ok(transaction_id)
    }

    pub fn add_operation(
        &self,
        transaction_id: TransactionId,
        operation: TransactionOperation,
    ) -> Result<()> {
        let mut active_txns = self.active_transactions.write();
        let transaction = active_txns
            .get_mut(&transaction_id)
            .ok_or_else(|| anyhow!("Transaction not found: {}", transaction_id))?;

        if transaction.status != TransactionStatus::Active {
            return Err(anyhow!("Transaction is not active"));
        }

        if transaction
            .start_time
            .elapsed()
            .expect("SystemTime should not go backwards")
            > transaction.timeout
        {
            transaction.status = TransactionStatus::Aborted;
            return Err(anyhow!("Transaction timeout"));
        }

        self.acquire_locks_for_operation(transaction_id, &operation)?;

        let serializable_op = self.convert_to_serializable(&operation);
        self.write_ahead_log
            .append(transaction_id, serializable_op)?;

        transaction.operations.push(operation);
        Ok(())
    }

    pub fn commit_transaction(&self, transaction_id: TransactionId) -> Result<()> {
        let mut active_txns = self.active_transactions.write();
        let transaction = active_txns
            .remove(&transaction_id)
            .ok_or_else(|| anyhow!("Transaction not found: {}", transaction_id))?;

        if transaction.status != TransactionStatus::Active {
            return Err(anyhow!("Transaction is not active"));
        }

        self.validate_transaction(&transaction)?;

        self.write_ahead_log.append(
            transaction_id,
            SerializableOperation::Commit { transaction_id },
        )?;

        for operation in &transaction.operations {
            self.execute_operation(operation)?;
        }

        self.lock_manager.release_transaction_locks(transaction_id);

        tracing::debug!("Transaction {} committed successfully", transaction_id);
        Ok(())
    }

    pub fn abort_transaction(&self, transaction_id: TransactionId) -> Result<()> {
        let mut active_txns = self.active_transactions.write();
        let _transaction = active_txns
            .remove(&transaction_id)
            .ok_or_else(|| anyhow!("Transaction not found: {}", transaction_id))?;

        self.write_ahead_log.append(
            transaction_id,
            SerializableOperation::Abort { transaction_id },
        )?;

        self.lock_manager.release_transaction_locks(transaction_id);

        tracing::debug!("Transaction {} aborted", transaction_id);
        Ok(())
    }

    fn acquire_locks_for_operation(
        &self,
        transaction_id: TransactionId,
        operation: &TransactionOperation,
    ) -> Result<()> {
        match operation {
            TransactionOperation::Insert { uri, .. } => {
                self.lock_manager
                    .acquire_lock(transaction_id, uri, LockType::Exclusive)?;
            }
            TransactionOperation::Update { uri, .. } => {
                self.lock_manager
                    .acquire_lock(transaction_id, uri, LockType::Exclusive)?;
            }
            TransactionOperation::Delete { uri, .. } => {
                self.lock_manager
                    .acquire_lock(transaction_id, uri, LockType::Exclusive)?;
            }
            TransactionOperation::BatchInsert { items } => {
                for (uri, _) in items {
                    self.lock_manager
                        .acquire_lock(transaction_id, uri, LockType::Exclusive)?;
                }
            }
            TransactionOperation::IndexRebuild { .. } => {
                self.lock_manager
                    .acquire_lock(transaction_id, "_global_", LockType::Exclusive)?;
            }
        }
        Ok(())
    }

    fn validate_transaction(&self, _transaction: &Transaction) -> Result<()> {
        Ok(())
    }

    fn execute_operation(&self, _operation: &TransactionOperation) -> Result<()> {
        Ok(())
    }

    fn convert_to_serializable(&self, operation: &TransactionOperation) -> SerializableOperation {
        match operation {
            TransactionOperation::Insert { uri, vector, .. } => SerializableOperation::Insert {
                uri: uri.clone(),
                vector_data: vector.as_f32(),
            },
            TransactionOperation::Update {
                uri,
                vector,
                old_vector,
            } => SerializableOperation::Update {
                uri: uri.clone(),
                new_vector: vector.as_f32(),
                old_vector: old_vector.as_ref().map(|v| v.as_f32()),
            },
            TransactionOperation::Delete { uri, .. } => {
                SerializableOperation::Delete { uri: uri.clone() }
            }
            _ => SerializableOperation::Insert {
                uri: "batch_operation".to_string(),
                vector_data: vec![0.0],
            },
        }
    }
}

impl IntegratedVectorStore {
    pub fn begin_transaction(&self, isolation_level: IsolationLevel) -> Result<TransactionId> {
        let transaction_id = self
            .transaction_manager
            .begin_transaction(isolation_level)?;
        self.metrics
            .active_transactions
            .fetch_add(1, Ordering::Relaxed);

        tracing::debug!("Started transaction {}", transaction_id);
        Ok(transaction_id)
    }

    pub fn commit_transaction(&self, transaction_id: TransactionId) -> Result<()> {
        self.transaction_manager
            .commit_transaction(transaction_id)?;
        self.metrics
            .active_transactions
            .fetch_sub(1, Ordering::Relaxed);
        self.metrics
            .committed_transactions
            .fetch_add(1, Ordering::Relaxed);

        tracing::debug!("Committed transaction {}", transaction_id);
        Ok(())
    }

    pub fn abort_transaction(&self, transaction_id: TransactionId) -> Result<()> {
        self.transaction_manager.abort_transaction(transaction_id)?;
        self.metrics
            .active_transactions
            .fetch_sub(1, Ordering::Relaxed);
        self.metrics
            .aborted_transactions
            .fetch_add(1, Ordering::Relaxed);

        tracing::debug!("Aborted transaction {}", transaction_id);
        Ok(())
    }

    pub fn transactional_insert(
        &self,
        transaction_id: TransactionId,
        uri: String,
        vector: crate::Vector,
        embedding_content: Option<EmbeddableContent>,
    ) -> Result<()> {
        if let Some(cached) = self.cache_manager.get_vector(&uri) {
            if cached.vector == vector {
                return Ok(());
            }
        }

        let operation = TransactionOperation::Insert {
            uri: uri.clone(),
            vector: vector.clone(),
            embedding_content,
        };

        self.transaction_manager
            .add_operation(transaction_id, operation)?;

        self.cache_manager.cache_vector(uri.clone(), vector.clone());

        let change_entry = ChangeLogEntry {
            id: self.generate_change_id(),
            timestamp: std::time::SystemTime::now(),
            operation: ChangeOperation::VectorInserted { uri, vector },
            metadata: HashMap::new(),
            transaction_id: Some(transaction_id),
        };
        self.change_log.add_entry(change_entry);

        self.metrics
            .total_operations
            .fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    pub fn stream_insert(
        &self,
        uri: String,
        vector: crate::Vector,
        priority: Priority,
    ) -> Result<()> {
        let operation = StreamingOperation::VectorInsert {
            uri,
            vector,
            priority,
        };
        self.streaming_engine.submit_operation(operation)?;
        Ok(())
    }

    pub fn batch_insert(
        &self,
        items: Vec<(String, crate::Vector)>,
        auto_commit: bool,
    ) -> Result<TransactionId> {
        let transaction_id = self.begin_transaction(IsolationLevel::ReadCommitted)?;

        for batch in items.chunks(self.config.batch_size) {
            let batch_operation = TransactionOperation::BatchInsert {
                items: batch.to_vec(),
            };
            self.transaction_manager
                .add_operation(transaction_id, batch_operation)?;
        }

        if auto_commit {
            self.commit_transaction(transaction_id)?;
        }

        Ok(transaction_id)
    }

    pub fn consistent_search(
        &self,
        query: &crate::Vector,
        k: usize,
        consistency_level: Option<ConsistencyLevel>,
    ) -> Result<Vec<(String, f32)>> {
        let effective_consistency = consistency_level.unwrap_or(self.config.consistency_level);

        let query_hash = self.compute_query_hash(query, k);
        if let Some(cached_result) = self.cache_manager.get_cached_query(&query_hash) {
            return Ok(cached_result.results);
        }

        match effective_consistency {
            ConsistencyLevel::Strong => {
                self.wait_for_consistency()?;
            }
            ConsistencyLevel::Session => {
                self.ensure_session_consistency()?;
            }
            ConsistencyLevel::Causal => {
                self.ensure_causal_consistency()?;
            }
            ConsistencyLevel::Eventual => {}
        }

        let store = self.vector_store.read();
        let results = store.similarity_search_vector(query, k)?;

        self.cache_manager
            .cache_query_result(query_hash, results.clone());

        Ok(results)
    }

    pub fn rdf_vector_search(
        &self,
        rdf_term: &str,
        k: usize,
        graph_context: Option<&str>,
    ) -> Result<Vec<(String, f32)>> {
        let rdf_integration = self.rdf_integration.read();

        let term = oxirs_core::model::Term::NamedNode(
            oxirs_core::model::NamedNode::new(rdf_term)
                .map_err(|e| anyhow!("Invalid IRI: {}", e))?,
        );

        let graph_name = graph_context
            .map(|ctx| -> Result<oxirs_core::model::GraphName> {
                Ok(oxirs_core::model::GraphName::NamedNode(
                    oxirs_core::model::NamedNode::new(ctx)
                        .map_err(|e| anyhow!("Invalid graph IRI: {}", e))?,
                ))
            })
            .transpose()?;

        let results = rdf_integration.find_similar_terms(&term, k, None, graph_name.as_ref())?;

        let converted_results = results
            .into_iter()
            .map(|result| (result.term.to_string(), result.score))
            .collect();

        Ok(converted_results)
    }

    pub fn sparql_vector_search(
        &self,
        _query: &str,
        bindings: &HashMap<String, String>,
    ) -> Result<Vec<HashMap<String, String>>> {
        let _sparql_service = self.sparql_service.read();
        Ok(vec![bindings.clone()])
    }

    pub fn subscribe_to_changes(
        &self,
        subscriber: std::sync::Arc<dyn ChangeSubscriber>,
    ) -> Result<()> {
        self.change_log.add_subscriber(subscriber);
        Ok(())
    }

    pub fn get_metrics(&self) -> StoreMetrics {
        StoreMetrics {
            total_vectors: AtomicU64::new(self.metrics.total_vectors.load(Ordering::Relaxed)),
            total_operations: AtomicU64::new(self.metrics.total_operations.load(Ordering::Relaxed)),
            successful_operations: AtomicU64::new(
                self.metrics.successful_operations.load(Ordering::Relaxed),
            ),
            failed_operations: AtomicU64::new(
                self.metrics.failed_operations.load(Ordering::Relaxed),
            ),
            average_operation_time_ms: std::sync::Arc::new(parking_lot::RwLock::new(
                *self.metrics.average_operation_time_ms.read(),
            )),
            active_transactions: AtomicU64::new(
                self.metrics.active_transactions.load(Ordering::Relaxed),
            ),
            committed_transactions: AtomicU64::new(
                self.metrics.committed_transactions.load(Ordering::Relaxed),
            ),
            aborted_transactions: AtomicU64::new(
                self.metrics.aborted_transactions.load(Ordering::Relaxed),
            ),
            replication_lag_ms: std::sync::Arc::new(parking_lot::RwLock::new(
                *self.metrics.replication_lag_ms.read(),
            )),
            consistency_violations: AtomicU64::new(
                self.metrics.consistency_violations.load(Ordering::Relaxed),
            ),
        }
    }

    pub fn health_check(&self) -> Result<HealthStatus> {
        let mut issues = Vec::new();

        if self.metrics.active_transactions.load(Ordering::Relaxed) > 1000 {
            issues.push("High number of active transactions".to_string());
        }

        let streaming_metrics = self.streaming_engine.get_metrics();
        if streaming_metrics.operations_pending.load(Ordering::Relaxed) > 10000 {
            issues.push("High number of pending streaming operations".to_string());
        }

        let cache_stats = self.cache_manager.get_stats();
        let hit_ratio = cache_stats.vector_cache_hits.load(Ordering::Relaxed) as f64
            / (cache_stats.vector_cache_hits.load(Ordering::Relaxed)
                + cache_stats.vector_cache_misses.load(Ordering::Relaxed)) as f64;

        if hit_ratio < 0.8 {
            issues.push("Low cache hit ratio".to_string());
        }

        if self.config.replication_config.enable_replication {
            let replication_lag = *self.metrics.replication_lag_ms.read();
            if replication_lag > 1000.0 {
                issues.push("High replication lag".to_string());
            }
        }

        let status = if issues.is_empty() {
            HealthStatus::Healthy
        } else if issues.len() <= 2 {
            HealthStatus::Warning(issues)
        } else {
            HealthStatus::Critical(issues)
        };

        Ok(status)
    }

    fn generate_change_id(&self) -> u64 {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        COUNTER.fetch_add(1, Ordering::Relaxed)
    }

    fn compute_query_hash(&self, query: &crate::Vector, k: usize) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        for value in query.as_f32() {
            value.to_bits().hash(&mut hasher);
        }
        k.hash(&mut hasher);
        hasher.finish()
    }

    fn wait_for_consistency(&self) -> Result<()> {
        let start = Instant::now();
        let timeout = Duration::from_secs(30);

        while self.metrics.active_transactions.load(Ordering::Relaxed) > 0 {
            if start.elapsed() > timeout {
                return Err(anyhow!("Timeout waiting for consistency"));
            }
            std::thread::sleep(Duration::from_millis(10));
        }

        Ok(())
    }

    fn ensure_session_consistency(&self) -> Result<()> {
        Ok(())
    }

    fn ensure_causal_consistency(&self) -> Result<()> {
        Ok(())
    }
}
