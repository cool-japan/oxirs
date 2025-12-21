//! # Store - accessors Methods
//!
//! This module contains method implementations for `Store`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;
use std::collections::{HashMap, HashSet};

impl Store {
    /// Get a reference to a dataset store (default if name is None)
    pub fn get_dataset(&self, name: Option<&str>) -> FusekiResult<Arc<RwLock<dyn CoreStore>>> {
        match name {
            None => Ok(Arc::clone(&self.default_store)),
            Some(dataset_name) => {
                let datasets = self
                    .datasets
                    .read()
                    .map_err(|e| FusekiError::store(format!("Failed to acquire read lock: {e}")))?;
                datasets
                    .get(dataset_name)
                    .map(Arc::clone)
                    .ok_or_else(|| FusekiError::not_found(format!("Dataset '{dataset_name}'")))
            }
        }
    }
    /// Execute a SPARQL query against a specific dataset
    pub fn query_dataset(
        &self,
        sparql: &str,
        dataset_name: Option<&str>,
    ) -> FusekiResult<QueryResult> {
        let start_time = Instant::now();
        debug!("Executing SPARQL query: {}", sparql.trim());
        let store = self.get_dataset(dataset_name)?;
        let store_guard = store
            .read()
            .map_err(|e| FusekiError::store(format!("Failed to acquire store read lock: {e}")))?;
        let core_result = self
            .query_engine
            .query(sparql, &*store_guard)
            .map_err(|e| FusekiError::query_execution(format!("Query execution failed: {e}")))?;
        let execution_time = start_time.elapsed();
        if let Ok(mut metadata) = self.metadata.write() {
            metadata.total_queries += 1;
            metadata.last_modified = Some(Instant::now());
        }
        let (result_count, query_type) = match &core_result {
            CoreQueryResult::Select { bindings, .. } => (bindings.len(), "SELECT"),
            CoreQueryResult::Construct(triples) => (triples.len(), "CONSTRUCT"),
            CoreQueryResult::Ask(_) => (1, "ASK"),
        };
        let query_stats = QueryStats {
            execution_time,
            result_count,
            query_type: query_type.to_string(),
            success: true,
            error_message: None,
        };
        debug!("Query executed successfully in {:?}", execution_time);
        Ok(QueryResult {
            inner: core_result,
            stats: query_stats,
        })
    }
    /// Execute a SPARQL update against a specific dataset
    pub fn update_dataset(
        &self,
        sparql: &str,
        dataset_name: Option<&str>,
    ) -> FusekiResult<UpdateResult> {
        let start_time = Instant::now();
        debug!("Executing SPARQL update: {}", sparql.trim());
        let store = self.get_dataset(dataset_name)?;
        let mut store_guard = store
            .write()
            .map_err(|e| FusekiError::store(format!("Failed to acquire store write lock: {e}")))?;
        let (operation_type, quads_inserted, quads_deleted, affected_graphs) =
            self.execute_sparql_update(&mut *store_guard, sparql)?;
        let execution_time = start_time.elapsed();
        if let Ok(mut metadata) = self.metadata.write() {
            metadata.total_updates += 1;
            metadata.last_modified = Some(Instant::now());
        }
        let update_stats = UpdateStats {
            execution_time,
            quads_inserted,
            quads_deleted,
            operation_type: operation_type.to_string(),
            success: true,
            error_message: None,
        };
        info!(
            "Update executed successfully in {:?}: {} quads inserted, {} quads deleted",
            execution_time, quads_inserted, quads_deleted
        );
        if let Ok(mut metadata) = self.metadata.write() {
            metadata.last_change_id += 1;
            let change = StoreChange {
                id: metadata.last_change_id,
                timestamp: chrono::Utc::now(),
                operation_type: operation_type.to_string(),
                affected_graphs,
                triple_count: quads_inserted + quads_deleted,
                dataset_name: dataset_name.map(|s| s.to_string()),
            };
            metadata.change_log.push(change);
            if metadata.change_log.len() > 1000 {
                let drain_end = metadata.change_log.len() - 1000;
                metadata.change_log.drain(0..drain_end);
            }
        }
        Ok(UpdateResult {
            stats: update_stats,
        })
    }
    /// Get store statistics
    pub fn get_stats(&self, dataset_name: Option<&str>) -> FusekiResult<StoreStats> {
        let store = self.get_dataset(dataset_name)?;
        let store_guard = store
            .read()
            .map_err(|e| FusekiError::store(format!("Failed to acquire store read lock: {e}")))?;
        let metadata = self.metadata.read().map_err(|e| {
            FusekiError::store(format!("Failed to acquire metadata read lock: {e}"))
        })?;
        let triple_count = store_guard
            .len()
            .map_err(|e| FusekiError::store(format!("Failed to get store size: {e}")))?;
        let uptime = metadata
            .created_at
            .map(|start| start.elapsed())
            .unwrap_or_default();
        let dataset_count = self
            .datasets
            .read()
            .map(|datasets| datasets.len())
            .unwrap_or(0);
        Ok(StoreStats {
            triple_count,
            dataset_count,
            total_queries: metadata.total_queries,
            total_updates: metadata.total_updates,
            cache_hit_ratio: if metadata.query_cache_hits + metadata.query_cache_misses > 0 {
                metadata.query_cache_hits as f64
                    / (metadata.query_cache_hits + metadata.query_cache_misses) as f64
            } else {
                0.0
            },
            uptime_seconds: uptime.as_secs(),
            change_log_size: metadata.change_log.len(),
            latest_change_id: metadata.last_change_id,
        })
    }
}
