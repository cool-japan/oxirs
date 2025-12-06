//! # Store - new_group Methods
//!
//! This module contains method implementations for `Store`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;
use std::collections::{HashMap, HashSet};

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
        if !path.exists() {
            std::fs::create_dir_all(path).map_err(|e| {
                FusekiError::store(format!("Failed to create store directory: {e}"))
            })?;
        }
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
    /// Extract unique graph names from quads
    pub(super) fn extract_graph_names(&self, quads: &[Quad]) -> Vec<String> {
        let mut graphs = std::collections::HashSet::new();
        for quad in quads {
            let graph_name = match quad.graph_name() {
                oxirs_core::model::GraphName::DefaultGraph => "default".to_string(),
                oxirs_core::model::GraphName::NamedNode(node) => node.as_str().to_string(),
                oxirs_core::model::GraphName::BlankNode(node) => {
                    format!("_:{}", node.as_str())
                }
                oxirs_core::model::GraphName::Variable(var) => {
                    format!("?{}", var.as_str())
                }
            };
            graphs.insert(graph_name);
        }
        let mut graph_vec: Vec<String> = graphs.into_iter().collect();
        graph_vec.sort();
        graph_vec
    }
    /// Clear a specific named graph
    pub(super) fn clear_named_graph(
        &self,
        store: &mut dyn CoreStore,
        graph_iri: &str,
    ) -> FusekiResult<(&'static str, usize, usize, Vec<String>)> {
        let named_node = oxirs_core::model::NamedNode::new(graph_iri).map_err(|e| {
            FusekiError::update_execution(format!("Invalid graph IRI '{graph_iri}': {e}"))
        })?;
        let graph_name = oxirs_core::model::GraphName::NamedNode(named_node);
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
        Ok(("CLEAR GRAPH", 0, deleted_count, vec![graph_iri.to_string()]))
    }
    /// Execute DELETE/INSERT operation (simplified implementation)
    pub(super) fn execute_delete_insert_operation(
        &self,
        store: &mut dyn CoreStore,
        sparql: &str,
    ) -> FusekiResult<(&'static str, usize, usize, Vec<String>)> {
        warn!("DELETE/INSERT operation using simplified implementation");
        let mut inserted_count = 0;
        let mut deleted_count = 0;
        let mut all_quads = Vec::new();
        if let Ok(delete_block) = self.extract_data_block(sparql, "DELETE") {
            let delete_quads = self.parse_data_block(&delete_block)?;
            all_quads.extend(delete_quads.iter().cloned());
            for quad in delete_quads {
                if store.remove_quad(&quad).map_err(|e| {
                    FusekiError::update_execution(format!("Failed to remove quad: {e}"))
                })? {
                    deleted_count += 1;
                }
            }
        }
        if let Ok(insert_block) = self.extract_data_block(sparql, "INSERT") {
            let insert_quads = self.parse_data_block(&insert_block)?;
            all_quads.extend(insert_quads.iter().cloned());
            for quad in insert_quads {
                if store.insert_quad(quad).map_err(|e| {
                    FusekiError::update_execution(format!("Failed to insert quad: {e}"))
                })? {
                    inserted_count += 1;
                }
            }
        }
        let affected_graphs = self.extract_graph_names(&all_quads);
        Ok((
            "DELETE/INSERT",
            inserted_count,
            deleted_count,
            affected_graphs,
        ))
    }
    /// Execute DELETE WHERE operation (simplified implementation)
    pub(super) fn execute_delete_where_operation(
        &self,
        store: &mut dyn CoreStore,
        sparql: &str,
    ) -> FusekiResult<(&'static str, usize, usize, Vec<String>)> {
        warn!(
            "DELETE WHERE operation using simplified implementation - deleting all matching patterns"
        );
        if let Ok(where_block) = self.extract_data_block(sparql, "WHERE") {
            let pattern_quads = self.parse_data_block(&where_block)?;
            let mut deleted_count = 0;
            let mut all_deleted_quads = Vec::new();
            for pattern_quad in pattern_quads {
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
                    all_deleted_quads.push(quad.clone());
                    if store.remove_quad(&quad).map_err(|e| {
                        FusekiError::update_execution(format!("Failed to remove quad: {e}"))
                    })? {
                        deleted_count += 1;
                    }
                }
            }
            let affected_graphs = self.extract_graph_names(&all_deleted_quads);
            Ok(("DELETE WHERE", 0, deleted_count, affected_graphs))
        } else {
            Err(FusekiError::update_execution(
                "Failed to extract WHERE block from DELETE WHERE operation".to_string(),
            ))
        }
    }
    /// Execute LOAD operation (SPARQL 1.1)
    /// Syntax: LOAD <sourceIRI> [INTO GRAPH <targetIRI>]
    pub(super) fn execute_load_operation(
        &self,
        store: &mut dyn CoreStore,
        sparql: &str,
    ) -> FusekiResult<(&'static str, usize, usize, Vec<String>)> {
        info!("Executing LOAD operation: {}", sparql.trim());
        let (source_iri, target_graph) = self.parse_load_statement(sparql)?;
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| FusekiError::update_execution(format!("Failed to create runtime: {e}")))?;
        let (data, content_type) =
            runtime.block_on(async { self.fetch_rdf_from_url(&source_iri).await })?;
        let format = self.detect_rdf_format(&source_iri, content_type.as_deref())?;
        let parser = Parser::new(format);
        let quads = parser.parse_str_to_quads(&data).map_err(|e| {
            FusekiError::parse(format!("Failed to parse RDF from '{}': {e}", source_iri))
        })?;
        let mut inserted_count = 0;
        let target_graph_name = if let Some(graph_iri) = target_graph {
            let named_node = oxirs_core::model::NamedNode::new(&graph_iri).map_err(|e| {
                FusekiError::update_execution(format!(
                    "Invalid target graph IRI '{graph_iri}': {e}"
                ))
            })?;
            Some(oxirs_core::model::GraphName::NamedNode(named_node))
        } else {
            None
        };
        for quad in quads {
            let final_quad = if let Some(ref target) = target_graph_name {
                oxirs_core::model::Quad::new(
                    quad.subject().clone(),
                    quad.predicate().clone(),
                    quad.object().clone(),
                    target.clone(),
                )
            } else {
                quad
            };
            if store
                .insert_quad(final_quad)
                .map_err(|e| FusekiError::update_execution(format!("Failed to insert quad: {e}")))?
            {
                inserted_count += 1;
            }
        }
        info!(
            "LOAD operation completed: loaded {} triples from '{}'",
            inserted_count, source_iri
        );
        let affected_graphs = if let Some(ref target_graph_name) = target_graph_name {
            match target_graph_name {
                oxirs_core::model::GraphName::DefaultGraph => vec!["default".to_string()],
                oxirs_core::model::GraphName::NamedNode(node) => {
                    vec![node.as_str().to_string()]
                }
                oxirs_core::model::GraphName::BlankNode(node) => {
                    vec![format!("_:{}", node.as_str())]
                }
                oxirs_core::model::GraphName::Variable(var) => {
                    vec![format!("?{}", var.as_str())]
                }
            }
        } else {
            vec!["default".to_string()]
        };
        Ok(("LOAD", inserted_count, 0, affected_graphs))
    }
    /// Execute COPY operation (SPARQL 1.1)
    /// Syntax: COPY [SILENT] (GRAPH <sourceIRI> | DEFAULT) TO (GRAPH <targetIRI> | DEFAULT)
    pub(super) fn execute_copy_operation(
        &self,
        store: &mut dyn CoreStore,
        sparql: &str,
    ) -> FusekiResult<(&'static str, usize, usize, Vec<String>)> {
        info!("Executing COPY operation: {}", sparql.trim());
        let sparql_upper = sparql.to_uppercase();
        let silent = sparql_upper.contains("SILENT");
        let (source_graph, target_graph) = self.parse_graph_management_statement(sparql, "COPY")?;
        self.clear_graph_by_name(store, &target_graph)?;
        let source_quads = self.get_quads_from_graph(store, &source_graph)?;
        let target_graph_name = self.graph_name_from_string(&target_graph)?;
        let mut copied_count = 0;
        for quad in &source_quads {
            let new_quad = oxirs_core::model::Quad::new(
                quad.subject().clone(),
                quad.predicate().clone(),
                quad.object().clone(),
                target_graph_name.clone(),
            );
            if store
                .insert_quad(new_quad)
                .map_err(|e| FusekiError::update_execution(format!("Failed to insert quad: {e}")))?
            {
                copied_count += 1;
            }
        }
        info!(
            "Copied {} quads from '{}' to '{}'",
            copied_count, source_graph, target_graph
        );
        if silent {
            Ok((
                "COPY SILENT",
                copied_count,
                0,
                vec![source_graph, target_graph],
            ))
        } else {
            Ok(("COPY", copied_count, 0, vec![source_graph, target_graph]))
        }
    }
    /// Execute MOVE operation (SPARQL 1.1)
    /// Syntax: MOVE [SILENT] (GRAPH <sourceIRI> | DEFAULT) TO (GRAPH <targetIRI> | DEFAULT)
    pub(super) fn execute_move_operation(
        &self,
        store: &mut dyn CoreStore,
        sparql: &str,
    ) -> FusekiResult<(&'static str, usize, usize, Vec<String>)> {
        info!("Executing MOVE operation: {}", sparql.trim());
        let sparql_upper = sparql.to_uppercase();
        let silent = sparql_upper.contains("SILENT");
        let (source_graph, target_graph) = self.parse_graph_management_statement(sparql, "MOVE")?;
        self.clear_graph_by_name(store, &target_graph)?;
        let source_quads = self.get_quads_from_graph(store, &source_graph)?;
        let target_graph_name = self.graph_name_from_string(&target_graph)?;
        let mut moved_count = 0;
        for quad in &source_quads {
            let new_quad = oxirs_core::model::Quad::new(
                quad.subject().clone(),
                quad.predicate().clone(),
                quad.object().clone(),
                target_graph_name.clone(),
            );
            if store
                .insert_quad(new_quad)
                .map_err(|e| FusekiError::update_execution(format!("Failed to insert quad: {e}")))?
            {
                moved_count += 1;
            }
        }
        for quad in source_quads {
            store.remove_quad(&quad).map_err(|e| {
                FusekiError::update_execution(format!("Failed to remove quad: {e}"))
            })?;
        }
        info!(
            "Moved {} quads from '{}' to '{}'",
            moved_count, source_graph, target_graph
        );
        if silent {
            Ok((
                "MOVE SILENT",
                moved_count,
                moved_count,
                vec![source_graph, target_graph],
            ))
        } else {
            Ok((
                "MOVE",
                moved_count,
                moved_count,
                vec![source_graph, target_graph],
            ))
        }
    }
    /// Execute ADD operation (SPARQL 1.1)
    /// Syntax: ADD [SILENT] (GRAPH <sourceIRI> | DEFAULT) TO (GRAPH <targetIRI> | DEFAULT)
    pub(super) fn execute_add_operation(
        &self,
        store: &mut dyn CoreStore,
        sparql: &str,
    ) -> FusekiResult<(&'static str, usize, usize, Vec<String>)> {
        info!("Executing ADD operation: {}", sparql.trim());
        let sparql_upper = sparql.to_uppercase();
        let silent = sparql_upper.contains("SILENT");
        let (source_graph, target_graph) = self.parse_graph_management_statement(sparql, "ADD")?;
        let source_quads = self.get_quads_from_graph(store, &source_graph)?;
        let target_graph_name = self.graph_name_from_string(&target_graph)?;
        let mut added_count = 0;
        for quad in &source_quads {
            let new_quad = oxirs_core::model::Quad::new(
                quad.subject().clone(),
                quad.predicate().clone(),
                quad.object().clone(),
                target_graph_name.clone(),
            );
            if store
                .insert_quad(new_quad)
                .map_err(|e| FusekiError::update_execution(format!("Failed to insert quad: {e}")))?
            {
                added_count += 1;
            }
        }
        info!(
            "Added {} quads from '{}' to '{}'",
            added_count, source_graph, target_graph
        );
        if silent {
            Ok((
                "ADD SILENT",
                added_count,
                0,
                vec![source_graph, target_graph],
            ))
        } else {
            Ok(("ADD", added_count, 0, vec![source_graph, target_graph]))
        }
    }
    /// Helper: Convert graph name string to GraphName
    pub(super) fn graph_name_from_string(
        &self,
        name: &str,
    ) -> FusekiResult<oxirs_core::model::GraphName> {
        if name == "default" {
            Ok(oxirs_core::model::GraphName::DefaultGraph)
        } else {
            let named_node = oxirs_core::model::NamedNode::new(name).map_err(|e| {
                FusekiError::update_execution(format!("Invalid graph IRI '{name}': {e}"))
            })?;
            Ok(oxirs_core::model::GraphName::NamedNode(named_node))
        }
    }
    /// Parse data block into quads using simple N-Triples-like parsing
    /// Handles both plain triples and GRAPH <iri> { triples } syntax
    pub(super) fn parse_data_block(&self, data_block: &str) -> FusekiResult<Vec<Quad>> {
        let mut quads = Vec::new();
        let data_block = data_block.trim();
        let data_upper = data_block.to_uppercase();
        if data_upper.starts_with("GRAPH") {
            let after_graph = &data_block[5..].trim_start();
            if let Some(open_bracket) = after_graph.find('<') {
                if let Some(close_bracket) = after_graph[open_bracket + 1..].find('>') {
                    let graph_iri =
                        &after_graph[open_bracket + 1..open_bracket + 1 + close_bracket];
                    if let Some(open_brace) = after_graph.find('{') {
                        if let Some(close_brace) = after_graph.rfind('}') {
                            let triples_block = &after_graph[open_brace + 1..close_brace].trim();
                            let graph_name =
                                oxirs_core::model::NamedNode::new(graph_iri).map_err(|e| {
                                    FusekiError::update_execution(format!("Invalid graph IRI: {e}"))
                                })?;
                            let graph_name_obj =
                                oxirs_core::model::GraphName::NamedNode(graph_name);
                            let parser = Parser::new(CoreRdfFormat::NTriples);
                            for line in triples_block.lines() {
                                let line = line.trim();
                                if line.is_empty() || line.starts_with('#') {
                                    continue;
                                }
                                let line_with_period = if line.ends_with('.') {
                                    line.to_string()
                                } else {
                                    format!("{line}.")
                                };
                                match parser.parse_str_to_quads(&line_with_period) {
                                    Ok(parsed_quads) => {
                                        for quad in parsed_quads {
                                            let new_quad = oxirs_core::model::Quad::new(
                                                quad.subject().clone(),
                                                quad.predicate().clone(),
                                                quad.object().clone(),
                                                graph_name_obj.clone(),
                                            );
                                            quads.push(new_quad);
                                        }
                                    }
                                    Err(e) => {
                                        warn!("Failed to parse line '{}': {}", line, e);
                                    }
                                }
                            }
                            return Ok(quads);
                        }
                    }
                }
            }
            return Err(FusekiError::update_execution(
                "Invalid GRAPH syntax in data block".to_string(),
            ));
        }
        let parser = Parser::new(CoreRdfFormat::NTriples);
        for line in data_block.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let line_with_period = if line.ends_with('.') {
                line.to_string()
            } else {
                format!("{line}.")
            };
            match parser.parse_str_to_quads(&line_with_period) {
                Ok(parsed_quads) => {
                    quads.extend(parsed_quads);
                }
                Err(e) => {
                    warn!("Failed to parse line '{}': {}", line, e);
                }
            }
        }
        Ok(quads)
    }
}
