//! Production-ready RDF store implementation integrating with oxirs-core

use crate::config::DatasetConfig;
use crate::error::{FusekiError, FusekiResult};
use oxirs_core::model::*;
use oxirs_core::parser::{Parser, RdfFormat as CoreRdfFormat};
use oxirs_core::query::{QueryEngine, QueryResult as CoreQueryResult};
use oxirs_core::serializer::Serializer;
use oxirs_core::{RdfStore, Store as CoreStore};
use serde::Serialize;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Type alias for dataset storage mapping
type DatasetMap = Arc<RwLock<HashMap<String, Arc<RwLock<dyn CoreStore>>>>>;

/// SPARQL query result formats
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResultFormat {
    Json,
    Xml,
    Csv,
    Tsv,
}

/// RDF serialization formats
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RdfSerializationFormat {
    Turtle,
    NTriples,
    RdfXml,
    JsonLd,
    NQuads,
}

/// Query execution statistics
#[derive(Debug, Clone, Serialize)]
pub struct QueryStats {
    pub execution_time: Duration,
    pub result_count: usize,
    pub query_type: String,
    pub success: bool,
    pub error_message: Option<String>,
}

/// Update execution statistics
#[derive(Debug, Clone, Serialize)]
pub struct UpdateStats {
    pub execution_time: Duration,
    pub quads_inserted: usize,
    pub quads_deleted: usize,
    pub operation_type: String,
    pub success: bool,
    pub error_message: Option<String>,
}

/// Multi-dataset store manager
#[derive(Clone)]
pub struct Store {
    /// Default dataset store
    pub(crate) default_store: Arc<RwLock<dyn CoreStore>>,
    /// Named datasets
    datasets: DatasetMap,
    /// Query engine for SPARQL execution
    query_engine: Arc<QueryEngine>,
    /// Store metadata
    metadata: Arc<RwLock<StoreMetadata>>,
}

impl std::fmt::Debug for Store {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Store")
            .field("default_store", &"CoreStore")
            .field("datasets", &"<datasets>")
            .field("query_engine", &"QueryEngine")
            .field("metadata", &self.metadata)
            .finish()
    }
}

/// Store metadata and statistics
#[derive(Debug, Clone, Default)]
struct StoreMetadata {
    created_at: Option<Instant>,
    last_modified: Option<Instant>,
    total_queries: u64,
    total_updates: u64,
    query_cache_hits: u64,
    query_cache_misses: u64,
    change_log: Vec<StoreChange>,
    last_change_id: u64,
}

/// Represents a change in the store for WebSocket notifications
#[derive(Debug, Clone, Serialize)]
pub struct StoreChange {
    pub id: u64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub operation_type: String,
    pub affected_graphs: Vec<String>,
    pub triple_count: usize,
    pub dataset_name: Option<String>,
}

/// Parameters for change detection
#[derive(Debug, Clone)]
pub struct ChangeDetectionParams {
    pub since: chrono::DateTime<chrono::Utc>,
    pub graphs: Option<Vec<String>>,
    pub operation_types: Option<Vec<String>>,
    pub limit: Option<usize>,
}

impl Store {
    /// Create a new in-memory store
    pub fn new() -> FusekiResult<Self> {
        let default_store = RdfStore::new()
            .map_err(|e| FusekiError::store(format!("Failed to create core store: {e}")))?;

        Ok(Store {
            default_store: Arc::new(RwLock::new(default_store)),
            datasets: Arc::new(RwLock::new(HashMap::new())),
            query_engine: Arc::new(QueryEngine::new()),
            metadata: Arc::new(RwLock::new(StoreMetadata {
                created_at: Some(Instant::now()),
                ..Default::default()
            })),
        })
    }

    /// Open a persistent store from a directory path
    pub fn open<P: AsRef<Path>>(path: P) -> FusekiResult<Self> {
        let path = path.as_ref();
        info!("Opening persistent store at: {:?}", path);

        // For now, create the directory if it doesn't exist
        if !path.exists() {
            std::fs::create_dir_all(path).map_err(|e| {
                FusekiError::store(format!("Failed to create store directory: {e}"))
            })?;
        }

        // Create the core store - currently will be in-memory until disk persistence is implemented
        let default_store = RdfStore::open(path.join("default.db"))
            .map_err(|e| FusekiError::store(format!("Failed to open core store: {e}")))?;

        Ok(Store {
            default_store: Arc::new(RwLock::new(default_store)),
            datasets: Arc::new(RwLock::new(HashMap::new())),
            query_engine: Arc::new(QueryEngine::new()),
            metadata: Arc::new(RwLock::new(StoreMetadata {
                created_at: Some(Instant::now()),
                ..Default::default()
            })),
        })
    }

    /// Create a named dataset
    pub fn create_dataset(&self, name: &str, config: DatasetConfig) -> FusekiResult<()> {
        let mut datasets = self
            .datasets
            .write()
            .map_err(|e| FusekiError::store(format!("Failed to acquire write lock: {e}")))?;

        if datasets.contains_key(name) {
            return Err(FusekiError::store(format!(
                "Dataset '{name}' already exists"
            )));
        }

        let store = if config.location.is_empty() {
            RdfStore::new()
        } else {
            RdfStore::open(&config.location)
        }
        .map_err(|e| FusekiError::store(format!("Failed to create dataset store: {e}")))?;

        datasets.insert(name.to_string(), Arc::new(RwLock::new(store)));
        info!("Created dataset: '{}'", name);
        Ok(())
    }

    /// Remove a named dataset
    pub fn remove_dataset(&self, name: &str) -> FusekiResult<bool> {
        let mut datasets = self
            .datasets
            .write()
            .map_err(|e| FusekiError::store(format!("Failed to acquire write lock: {e}")))?;

        let removed = datasets.remove(name).is_some();
        if removed {
            info!("Removed dataset: '{}'", name);
        }
        Ok(removed)
    }

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

    /// List all dataset names
    pub fn list_datasets(&self) -> FusekiResult<Vec<String>> {
        let datasets = self
            .datasets
            .read()
            .map_err(|e| FusekiError::store(format!("Failed to acquire read lock: {e}")))?;

        Ok(datasets.keys().cloned().collect())
    }

    /// Check if the store is ready for operations
    pub fn is_ready(&self) -> bool {
        // Check if we can acquire locks without blocking
        self.default_store.try_read().is_ok()
            && self.datasets.try_read().is_ok()
            && self.metadata.try_read().is_ok()
    }

    /// Execute a SPARQL query against the default dataset
    pub fn query(&self, sparql: &str) -> FusekiResult<QueryResult> {
        self.query_dataset(sparql, None)
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

        // Execute the query using the query engine
        let core_result = self
            .query_engine
            .query(sparql, &*store_guard)
            .map_err(|e| FusekiError::query_execution(format!("Query execution failed: {e}")))?;

        let execution_time = start_time.elapsed();

        // Update metadata
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

    /// Execute a SPARQL update against the default dataset
    pub fn update(&self, sparql: &str) -> FusekiResult<UpdateResult> {
        self.update_dataset(sparql, None)
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

        // Parse and execute the update operation
        let (operation_type, quads_inserted, quads_deleted) =
            self.execute_sparql_update(&mut *store_guard, sparql)?;

        let execution_time = start_time.elapsed();

        // Update metadata
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

        // Record the change for WebSocket notifications
        if let Ok(mut metadata) = self.metadata.write() {
            metadata.last_change_id += 1;
            let change = StoreChange {
                id: metadata.last_change_id,
                timestamp: chrono::Utc::now(),
                operation_type: operation_type.to_string(),
                affected_graphs: vec!["default".to_string()], // TODO: extract actual graphs
                triple_count: quads_inserted + quads_deleted,
                dataset_name: dataset_name.map(|s| s.to_string()),
            };

            metadata.change_log.push(change);

            // Keep only the last 1000 changes to prevent memory issues
            if metadata.change_log.len() > 1000 {
                let drain_end = metadata.change_log.len() - 1000;
                metadata.change_log.drain(0..drain_end);
            }
        }

        Ok(UpdateResult {
            stats: update_stats,
        })
    }

    /// Execute a SPARQL update operation using basic parsing and low-level Store API
    fn execute_sparql_update(
        &self,
        store: &mut dyn CoreStore,
        sparql: &str,
    ) -> FusekiResult<(&'static str, usize, usize)> {
        let sparql_upper = sparql.to_uppercase();

        if sparql_upper.contains("CLEAR") {
            self.execute_clear_operation(store, sparql)
        } else if sparql_upper.contains("INSERT DATA") {
            self.execute_insert_data_operation(store, sparql)
        } else if sparql_upper.contains("DELETE DATA") {
            self.execute_delete_data_operation(store, sparql)
        } else if sparql_upper.contains("DELETE") && sparql_upper.contains("INSERT") {
            self.execute_delete_insert_operation(store, sparql)
        } else if sparql_upper.contains("DELETE") && sparql_upper.contains("WHERE") {
            self.execute_delete_where_operation(store, sparql)
        } else if sparql_upper.contains("INSERT") {
            self.execute_insert_operation(store, sparql)
        } else if sparql_upper.contains("LOAD") {
            self.execute_load_operation(store, sparql)
        } else {
            warn!(
                "SPARQL update operation not recognized or supported: {}",
                sparql.trim()
            );
            Ok(("UNKNOWN", 0, 0))
        }
    }

    /// Execute CLEAR operation
    fn execute_clear_operation(
        &self,
        store: &mut dyn CoreStore,
        sparql: &str,
    ) -> FusekiResult<(&'static str, usize, usize)> {
        let sparql_upper = sparql.to_uppercase();

        // Check if clearing all graphs
        if sparql_upper.contains("CLEAR ALL") {
            let quad_count = store.len().map_err(|e| {
                FusekiError::update_execution(format!("Failed to get store size: {e}"))
            })?;

            store.clear_all().map_err(|e| {
                FusekiError::update_execution(format!("Failed to clear all graphs: {e}"))
            })?;

            info!("Cleared all graphs: {} quads removed", quad_count);
            return Ok(("CLEAR ALL", 0, quad_count));
        }

        // Check if clearing default graph
        if sparql_upper.contains("CLEAR DEFAULT") || !sparql_upper.contains("CLEAR GRAPH <") {
            return self.clear_default_graph(store);
        }

        // Extract named graph IRI for specific graph clearing
        if let Some(graph_iri) = self.extract_graph_iri(sparql)? {
            return self.clear_named_graph(store, &graph_iri);
        }

        // If no specific graph identified, clear default graph
        self.clear_default_graph(store)
    }

    /// Clear the default graph
    fn clear_default_graph(
        &self,
        store: &mut dyn CoreStore,
    ) -> FusekiResult<(&'static str, usize, usize)> {
        // Get all quads in the default graph
        let default_quads = store
            .find_quads(
                None,
                None,
                None,
                Some(&oxirs_core::model::GraphName::DefaultGraph),
            )
            .map_err(|e| {
                FusekiError::update_execution(format!("Failed to query default graph: {e}"))
            })?;

        let mut deleted_count = 0;
        for quad in default_quads {
            if store.remove_quad(&quad).map_err(|e| {
                FusekiError::update_execution(format!(
                    "Failed to remove quad from default graph: {e}"
                ))
            })? {
                deleted_count += 1;
            }
        }

        info!("Cleared default graph: {} quads removed", deleted_count);
        Ok(("CLEAR DEFAULT", 0, deleted_count))
    }

    /// Clear a specific named graph
    fn clear_named_graph(
        &self,
        store: &mut dyn CoreStore,
        graph_iri: &str,
    ) -> FusekiResult<(&'static str, usize, usize)> {
        // Create the named graph reference
        let named_node = oxirs_core::model::NamedNode::new(graph_iri).map_err(|e| {
            FusekiError::update_execution(format!("Invalid graph IRI '{graph_iri}': {e}"))
        })?;
        let graph_name = oxirs_core::model::GraphName::NamedNode(named_node);

        // Get all quads in the specified named graph
        let graph_quads = store
            .find_quads(None, None, None, Some(&graph_name))
            .map_err(|e| {
                FusekiError::update_execution(format!(
                    "Failed to query named graph '{graph_iri}': {e}"
                ))
            })?;

        let mut deleted_count = 0;
        for quad in graph_quads {
            if store.remove_quad(&quad).map_err(|e| {
                FusekiError::update_execution(format!(
                    "Failed to remove quad from named graph '{graph_iri}': {e}"
                ))
            })? {
                deleted_count += 1;
            }
        }

        info!(
            "Cleared named graph '{}': {} quads removed",
            graph_iri, deleted_count
        );
        Ok(("CLEAR GRAPH", 0, deleted_count))
    }

    /// Extract graph IRI from CLEAR GRAPH statement
    fn extract_graph_iri(&self, sparql: &str) -> FusekiResult<Option<String>> {
        let sparql_upper = sparql.to_uppercase();

        // Look for "CLEAR GRAPH <IRI>" pattern
        if let Some(start_pos) = sparql_upper.find("CLEAR GRAPH") {
            let remaining = &sparql[start_pos + "CLEAR GRAPH".len()..];

            // Find opening angle bracket
            if let Some(open_bracket) = remaining.find('<') {
                // Find closing angle bracket
                if let Some(close_bracket) = remaining[open_bracket + 1..].find('>') {
                    let iri = &remaining[open_bracket + 1..open_bracket + 1 + close_bracket];
                    return Ok(Some(iri.trim().to_string()));
                }
            }

            // Also check for prefixed names (simplified)
            let tokens: Vec<&str> = remaining.split_whitespace().collect();
            if let Some(first_token) = tokens.first() {
                if !first_token.starts_with('<') && first_token.contains(':') {
                    // This might be a prefixed name like "ex:graph"
                    // For now, treat it as a simple IRI (proper prefix resolution would be more complex)
                    warn!("Prefixed graph names not fully supported yet: {first_token}");
                    return Ok(Some(format!("urn:x-local:{first_token}")));
                }
            }
        }

        Ok(None)
    }

    /// Execute INSERT DATA operation
    fn execute_insert_data_operation(
        &self,
        store: &mut dyn CoreStore,
        sparql: &str,
    ) -> FusekiResult<(&'static str, usize, usize)> {
        // Extract the data block between { and }
        let data_block = self.extract_data_block(sparql, "INSERT DATA")?;
        let quads = self.parse_data_block(&data_block)?;

        let mut inserted_count = 0;
        for quad in quads {
            if store
                .insert_quad(quad)
                .map_err(|e| FusekiError::update_execution(format!("Failed to insert quad: {e}")))?
            {
                inserted_count += 1;
            }
        }

        Ok(("INSERT DATA", inserted_count, 0))
    }

    /// Execute DELETE DATA operation  
    fn execute_delete_data_operation(
        &self,
        store: &mut dyn CoreStore,
        sparql: &str,
    ) -> FusekiResult<(&'static str, usize, usize)> {
        // Extract the data block between { and }
        let data_block = self.extract_data_block(sparql, "DELETE DATA")?;
        let quads = self.parse_data_block(&data_block)?;

        let mut deleted_count = 0;
        for quad in quads {
            if store
                .remove_quad(&quad)
                .map_err(|e| FusekiError::update_execution(format!("Failed to remove quad: {e}")))?
            {
                deleted_count += 1;
            }
        }

        Ok(("DELETE DATA", 0, deleted_count))
    }

    /// Execute DELETE/INSERT operation (simplified implementation)
    fn execute_delete_insert_operation(
        &self,
        store: &mut dyn CoreStore,
        sparql: &str,
    ) -> FusekiResult<(&'static str, usize, usize)> {
        // This is a simplified implementation - just parse both blocks and execute them
        warn!("DELETE/INSERT operation using simplified implementation");

        let mut inserted_count = 0;
        let mut deleted_count = 0;

        // Try to extract DELETE block
        if let Ok(delete_block) = self.extract_data_block(sparql, "DELETE") {
            let delete_quads = self.parse_data_block(&delete_block)?;
            for quad in delete_quads {
                if store.remove_quad(&quad).map_err(|e| {
                    FusekiError::update_execution(format!("Failed to remove quad: {e}"))
                })? {
                    deleted_count += 1;
                }
            }
        }

        // Try to extract INSERT block
        if let Ok(insert_block) = self.extract_data_block(sparql, "INSERT") {
            let insert_quads = self.parse_data_block(&insert_block)?;
            for quad in insert_quads {
                if store.insert_quad(quad).map_err(|e| {
                    FusekiError::update_execution(format!("Failed to insert quad: {e}"))
                })? {
                    inserted_count += 1;
                }
            }
        }

        Ok(("DELETE/INSERT", inserted_count, deleted_count))
    }

    /// Execute DELETE WHERE operation (simplified implementation)
    fn execute_delete_where_operation(
        &self,
        store: &mut dyn CoreStore,
        sparql: &str,
    ) -> FusekiResult<(&'static str, usize, usize)> {
        warn!("DELETE WHERE operation using simplified implementation - deleting all matching patterns");

        // For now, just extract the WHERE block and delete matching quads
        if let Ok(where_block) = self.extract_data_block(sparql, "WHERE") {
            let pattern_quads = self.parse_data_block(&where_block)?;
            let mut deleted_count = 0;

            for pattern_quad in pattern_quads {
                // Find all matching quads and delete them
                let matching_quads = store
                    .find_quads(
                        Some(pattern_quad.subject()),
                        Some(pattern_quad.predicate()),
                        Some(pattern_quad.object()),
                        Some(pattern_quad.graph_name()),
                    )
                    .map_err(|e| {
                        FusekiError::update_execution(format!(
                            "Failed to query matching quads: {e}"
                        ))
                    })?;

                for quad in matching_quads {
                    if store.remove_quad(&quad).map_err(|e| {
                        FusekiError::update_execution(format!("Failed to remove quad: {e}"))
                    })? {
                        deleted_count += 1;
                    }
                }
            }

            Ok(("DELETE WHERE", 0, deleted_count))
        } else {
            Err(FusekiError::update_execution(
                "Failed to extract WHERE block from DELETE WHERE operation".to_string(),
            ))
        }
    }

    /// Execute simple INSERT operation
    fn execute_insert_operation(
        &self,
        store: &mut dyn CoreStore,
        sparql: &str,
    ) -> FusekiResult<(&'static str, usize, usize)> {
        // Extract the data block and insert
        let data_block = self.extract_data_block(sparql, "INSERT")?;
        let quads = self.parse_data_block(&data_block)?;

        let mut inserted_count = 0;
        for quad in quads {
            if store
                .insert_quad(quad)
                .map_err(|e| FusekiError::update_execution(format!("Failed to insert quad: {e}")))?
            {
                inserted_count += 1;
            }
        }

        Ok(("INSERT", inserted_count, 0))
    }

    /// Execute LOAD operation (simplified implementation)
    fn execute_load_operation(
        &self,
        _store: &mut dyn CoreStore,
        sparql: &str,
    ) -> FusekiResult<(&'static str, usize, usize)> {
        warn!("LOAD operation not fully implemented: {}", sparql.trim());
        // TODO: Implement LOAD operation to fetch RDF from URL and insert into store
        Ok(("LOAD", 0, 0))
    }

    /// Extract data block from SPARQL update (between { and })
    fn extract_data_block(&self, sparql: &str, operation: &str) -> FusekiResult<String> {
        let operation_upper = operation.to_uppercase();
        let sparql_upper = sparql.to_uppercase();

        // Find the operation keyword
        let operation_pos = sparql_upper.find(&operation_upper).ok_or_else(|| {
            FusekiError::update_execution(format!("Operation '{operation}' not found"))
        })?;

        // Find the opening brace after the operation
        let remaining = &sparql[operation_pos + operation.len()..];
        let open_brace_pos = remaining.find('{').ok_or_else(|| {
            FusekiError::update_execution("Opening brace '{' not found".to_string())
        })?;

        // Find the matching closing brace
        let mut brace_count = 0;
        let mut close_brace_pos = None;
        let chars: Vec<char> = remaining.chars().collect();

        for (i, &ch) in chars.iter().enumerate().skip(open_brace_pos) {
            match ch {
                '{' => brace_count += 1,
                '}' => {
                    brace_count -= 1;
                    if brace_count == 0 {
                        close_brace_pos = Some(i);
                        break;
                    }
                }
                _ => {}
            }
        }

        let close_brace_pos = close_brace_pos.ok_or_else(|| {
            FusekiError::update_execution("Matching closing brace '}' not found".to_string())
        })?;

        // Extract the content between braces
        let data_block = &remaining[open_brace_pos + 1..close_brace_pos];
        Ok(data_block.trim().to_string())
    }

    /// Parse data block into quads using simple N-Triples-like parsing
    fn parse_data_block(&self, data_block: &str) -> FusekiResult<Vec<Quad>> {
        let mut quads = Vec::new();

        // Use the oxirs-core parser for N-Triples format
        let parser = Parser::new(CoreRdfFormat::NTriples);

        // Parse each line as a potential triple
        for line in data_block.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Add a period if missing (for simpler parsing)
            let line_with_period = if line.ends_with('.') {
                line.to_string()
            } else {
                format!("{line}.")
            };

            // Try to parse as N-Triples
            match parser.parse_str_to_quads(&line_with_period) {
                Ok(parsed_quads) => {
                    quads.extend(parsed_quads);
                }
                Err(e) => {
                    warn!("Failed to parse line '{}': {}", line, e);
                    // Continue parsing other lines
                }
            }
        }

        Ok(quads)
    }

    /// Load RDF data from a string into a dataset
    pub fn load_data(
        &self,
        data: &str,
        format: RdfSerializationFormat,
        dataset_name: Option<&str>,
    ) -> FusekiResult<usize> {
        let store = self.get_dataset(dataset_name)?;
        let store_guard = store
            .write()
            .map_err(|e| FusekiError::store(format!("Failed to acquire store write lock: {e}")))?;

        let core_format = match format {
            RdfSerializationFormat::Turtle => CoreRdfFormat::Turtle,
            RdfSerializationFormat::NTriples => CoreRdfFormat::NTriples,
            RdfSerializationFormat::RdfXml => CoreRdfFormat::RdfXml,
            RdfSerializationFormat::JsonLd => {
                return Err(FusekiError::unsupported_media_type(
                    "JSON-LD not supported yet",
                ))
            }
            RdfSerializationFormat::NQuads => CoreRdfFormat::NQuads,
        };

        let parser = Parser::new(core_format);
        let quads = parser
            .parse_str_to_quads(data)
            .map_err(|e| FusekiError::parse(format!("Failed to parse RDF data: {e}")))?;
        let graph = Graph::from_iter(
            quads
                .into_iter()
                .filter(|q| q.is_default_graph())
                .map(|q| q.to_triple()),
        );

        let mut inserted_count = 0;
        for triple in graph.iter() {
            if store_guard
                .insert_triple(triple.clone())
                .map_err(|e| FusekiError::store(format!("Failed to insert triple: {e}")))?
            {
                inserted_count += 1;
            }
        }

        info!("Loaded {} triples into dataset", inserted_count);
        Ok(inserted_count)
    }

    /// Export RDF data from a dataset as a string
    pub fn export_data(
        &self,
        format: RdfSerializationFormat,
        dataset_name: Option<&str>,
    ) -> FusekiResult<String> {
        let store = self.get_dataset(dataset_name)?;
        let store_guard = store
            .read()
            .map_err(|e| FusekiError::store(format!("Failed to acquire store read lock: {e}")))?;

        let core_format = match format {
            RdfSerializationFormat::Turtle => CoreRdfFormat::Turtle,
            RdfSerializationFormat::NTriples => CoreRdfFormat::NTriples,
            RdfSerializationFormat::RdfXml => CoreRdfFormat::RdfXml,
            RdfSerializationFormat::JsonLd => {
                return Err(FusekiError::unsupported_media_type(
                    "JSON-LD not supported yet",
                ))
            }
            RdfSerializationFormat::NQuads => CoreRdfFormat::NQuads,
        };

        let serializer = Serializer::new(core_format);
        let triples = store_guard
            .triples()
            .map_err(|e| FusekiError::store(format!("Failed to query triples: {e}")))?;

        let graph = Graph::from_triples(triples);
        let serialized = serializer
            .serialize_graph(&graph)
            .map_err(|e| FusekiError::parse(format!("Failed to serialize data: {e}")))?;

        Ok(serialized)
    }

    /// Get changes since a specific timestamp for WebSocket notifications
    pub async fn get_changes_since(
        &self,
        since: chrono::DateTime<chrono::Utc>,
    ) -> FusekiResult<Vec<StoreChange>> {
        let metadata = self.metadata.read().map_err(|e| {
            FusekiError::store(format!("Failed to acquire metadata read lock: {e}"))
        })?;

        let recent_changes = metadata
            .change_log
            .iter()
            .filter(|change| change.timestamp > since)
            .cloned()
            .collect();

        Ok(recent_changes)
    }

    /// Get changes with advanced filtering
    pub async fn get_changes_filtered(
        &self,
        params: ChangeDetectionParams,
    ) -> FusekiResult<Vec<StoreChange>> {
        let metadata = self.metadata.read().map_err(|e| {
            FusekiError::store(format!("Failed to acquire metadata read lock: {e}"))
        })?;

        let mut changes: Vec<StoreChange> = metadata
            .change_log
            .iter()
            .filter(|change| {
                // Filter by timestamp
                if change.timestamp <= params.since {
                    return false;
                }

                // Filter by graphs if specified
                if let Some(ref graphs) = params.graphs {
                    if !change.affected_graphs.iter().any(|g| graphs.contains(g)) {
                        return false;
                    }
                }

                // Filter by operation types if specified
                if let Some(ref op_types) = params.operation_types {
                    if !op_types.contains(&change.operation_type) {
                        return false;
                    }
                }

                true
            })
            .cloned()
            .collect();

        // Apply limit if specified
        if let Some(limit) = params.limit {
            if changes.len() > limit {
                changes.truncate(limit);
            }
        }

        Ok(changes)
    }

    /// Clear old changes from the log
    pub async fn cleanup_old_changes(
        &self,
        older_than: chrono::DateTime<chrono::Utc>,
    ) -> FusekiResult<usize> {
        let mut metadata = self.metadata.write().map_err(|e| {
            FusekiError::store(format!("Failed to acquire metadata write lock: {e}"))
        })?;

        let initial_count = metadata.change_log.len();
        metadata
            .change_log
            .retain(|change| change.timestamp > older_than);
        let removed_count = initial_count - metadata.change_log.len();

        if removed_count > 0 {
            debug!("Cleaned up {} old change log entries", removed_count);
        }

        Ok(removed_count)
    }

    /// Get the latest change ID
    pub async fn get_latest_change_id(&self) -> FusekiResult<u64> {
        let metadata = self.metadata.read().map_err(|e| {
            FusekiError::store(format!("Failed to acquire metadata read lock: {e}"))
        })?;

        Ok(metadata.last_change_id)
    }

    /// Watch for changes (used by WebSocket subscriptions)
    pub async fn watch_changes(&self, since_id: u64) -> FusekiResult<Vec<StoreChange>> {
        let metadata = self.metadata.read().map_err(|e| {
            FusekiError::store(format!("Failed to acquire metadata read lock: {e}"))
        })?;

        let new_changes = metadata
            .change_log
            .iter()
            .filter(|change| change.id > since_id)
            .cloned()
            .collect();

        Ok(new_changes)
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

impl Default for Store {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

/// Query result wrapper with statistics
#[derive(Debug)]
pub struct QueryResult {
    pub inner: CoreQueryResult,
    pub stats: QueryStats,
}

impl QueryResult {
    /// Convert to JSON string based on result type
    pub fn to_json(&self) -> FusekiResult<String> {
        match &self.inner {
            CoreQueryResult::Select {
                variables,
                bindings,
            } => {
                let json_result = serde_json::json!({
                    "head": {
                        "vars": variables
                    },
                    "results": {
                        "bindings": bindings.iter().map(|binding| {
                            binding.iter().map(|(k, v)| {
                                (k.clone(), serde_json::json!({
                                    "type": match v {
                                        Term::NamedNode(_) => "uri",
                                        Term::BlankNode(_) => "bnode",
                                        Term::Literal(_) => "literal",
                                        Term::Variable(_) => "variable",
                                        Term::QuotedTriple(_) => "quotedTriple",
                                    },
                                    "value": v.to_string()
                                }))
                            }).collect::<serde_json::Map<String, serde_json::Value>>()
                        }).collect::<Vec<_>>()
                    }
                });
                Ok(json_result.to_string())
            }
            CoreQueryResult::Ask(result) => {
                let json_result = serde_json::json!({
                    "head": {},
                    "boolean": result
                });
                Ok(json_result.to_string())
            }
            CoreQueryResult::Construct(_) => Err(FusekiError::unsupported_media_type(
                "CONSTRUCT queries should use RDF format, not JSON",
            )),
        }
    }

    /// Convert to XML string  
    pub fn to_xml(&self) -> FusekiResult<String> {
        match &self.inner {
            CoreQueryResult::Select {
                variables,
                bindings,
            } => {
                let mut xml = String::from("<?xml version=\"1.0\"?>\n<sparql xmlns=\"http://www.w3.org/2005/sparql-results#\">\n");
                xml.push_str("  <head>\n");
                for var in variables {
                    xml.push_str(&format!("    <variable name=\"{var}\"/>\n"));
                }
                xml.push_str("  </head>\n  <results>\n");

                for binding in bindings {
                    xml.push_str("    <result>\n");
                    for (var, term) in binding {
                        xml.push_str(&format!("      <binding name=\"{var}\">\n"));
                        match term {
                            Term::NamedNode(node) => {
                                xml.push_str(&format!("        <uri>{}</uri>\n", node.as_str()));
                            }
                            Term::BlankNode(node) => {
                                xml.push_str(&format!(
                                    "        <bnode>{}</bnode>\n",
                                    node.as_str()
                                ));
                            }
                            Term::Literal(literal) => {
                                xml.push_str(&format!(
                                    "        <literal>{}</literal>\n",
                                    literal.value()
                                ));
                            }
                            Term::Variable(variable) => {
                                xml.push_str(&format!(
                                    "        <variable>{}</variable>\n",
                                    variable.as_str()
                                ));
                            }
                            Term::QuotedTriple(triple) => {
                                xml.push_str(&format!(
                                    "        <quotedTriple>&lt;&lt;{} {} {}&gt;&gt;</quotedTriple>\n",
                                    triple.subject(), triple.predicate(), triple.object()
                                ));
                            }
                        }
                        xml.push_str("      </binding>\n");
                    }
                    xml.push_str("    </result>\n");
                }
                xml.push_str("  </results>\n</sparql>");
                Ok(xml)
            }
            CoreQueryResult::Ask(result) => {
                let xml = format!("<?xml version=\"1.0\"?>\n<sparql xmlns=\"http://www.w3.org/2005/sparql-results#\">\n  <head/>\n  <boolean>{result}</boolean>\n</sparql>");
                Ok(xml)
            }
            _ => Err(FusekiError::unsupported_media_type(
                "XML format only supported for SELECT and ASK queries",
            )),
        }
    }

    /// Convert to CSV string
    pub fn to_csv(&self) -> FusekiResult<String> {
        match &self.inner {
            CoreQueryResult::Select {
                variables,
                bindings,
            } => {
                let mut csv = variables.join(",");
                csv.push('\n');

                for binding in bindings {
                    let values: Vec<String> = variables
                        .iter()
                        .map(|var| {
                            binding
                                .get(var)
                                .map(|term| term.to_string())
                                .unwrap_or_default()
                        })
                        .collect();
                    csv.push_str(&values.join(","));
                    csv.push('\n');
                }

                Ok(csv)
            }
            _ => Err(FusekiError::unsupported_media_type(
                "CSV format only supported for SELECT queries",
            )),
        }
    }

    /// Convert to TSV string
    pub fn to_tsv(&self) -> FusekiResult<String> {
        match &self.inner {
            CoreQueryResult::Select {
                variables,
                bindings,
            } => {
                let mut tsv = variables.join("\t");
                tsv.push('\n');

                for binding in bindings {
                    let values: Vec<String> = variables
                        .iter()
                        .map(|var| {
                            binding
                                .get(var)
                                .map(|term| term.to_string())
                                .unwrap_or_default()
                        })
                        .collect();
                    tsv.push_str(&values.join("\t"));
                    tsv.push('\n');
                }

                Ok(tsv)
            }
            _ => Err(FusekiError::unsupported_media_type(
                "TSV format only supported for SELECT queries",
            )),
        }
    }

    /// Convert to RDF string (for CONSTRUCT/DESCRIBE)
    pub fn to_rdf(&self, format: RdfSerializationFormat) -> FusekiResult<String> {
        match &self.inner {
            CoreQueryResult::Construct(triples) => {
                let core_format = match format {
                    RdfSerializationFormat::Turtle => CoreRdfFormat::Turtle,
                    RdfSerializationFormat::NTriples => CoreRdfFormat::NTriples,
                    RdfSerializationFormat::RdfXml => CoreRdfFormat::RdfXml,
                    RdfSerializationFormat::JsonLd => {
                        return Err(FusekiError::unsupported_media_type(
                            "JSON-LD not supported yet",
                        ))
                    }
                    RdfSerializationFormat::NQuads => CoreRdfFormat::NQuads,
                };

                let serializer = Serializer::new(core_format);
                let graph = oxirs_core::model::graph::Graph::from_iter(triples.clone());
                serializer.serialize_graph(&graph).map_err(|e| {
                    FusekiError::parse(format!("Failed to serialize CONSTRUCT result: {e}"))
                })
            }
            _ => Err(FusekiError::unsupported_media_type(
                "RDF format only supported for CONSTRUCT and DESCRIBE queries",
            )),
        }
    }

    /// Get result in the specified format
    pub fn format_as(&self, format: ResultFormat) -> FusekiResult<String> {
        match format {
            ResultFormat::Json => self.to_json(),
            ResultFormat::Xml => self.to_xml(),
            ResultFormat::Csv => self.to_csv(),
            ResultFormat::Tsv => self.to_tsv(),
        }
    }
}

/// Update result wrapper with statistics
#[derive(Debug)]
pub struct UpdateResult {
    pub stats: UpdateStats,
}

/// Store statistics
#[derive(Debug, Serialize)]
pub struct StoreStats {
    pub triple_count: usize,
    pub dataset_count: usize,
    pub total_queries: u64,
    pub total_updates: u64,
    pub cache_hit_ratio: f64,
    pub uptime_seconds: u64,
    pub change_log_size: usize,
    pub latest_change_id: u64,
}
