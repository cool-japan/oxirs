//! # Store - execute_sparql_update_group Methods
//!
//! This module contains method implementations for `Store`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;
use std::collections::{HashMap, HashSet};

impl Store {
    /// Execute a SPARQL update operation using basic parsing and low-level Store API
    /// Returns: (operation_type, quads_inserted, quads_deleted, affected_graphs)
    pub(super) fn execute_sparql_update(
        &self,
        store: &mut dyn CoreStore,
        sparql: &str,
    ) -> FusekiResult<(&'static str, usize, usize, Vec<String>)> {
        let sparql_upper = sparql.to_uppercase();
        if sparql_upper.contains("CREATE") {
            self.execute_create_operation(store, sparql)
        } else if sparql_upper.contains("DROP") {
            self.execute_drop_operation(store, sparql)
        } else if sparql_upper.contains("COPY") {
            self.execute_copy_operation(store, sparql)
        } else if sparql_upper.contains("MOVE") {
            self.execute_move_operation(store, sparql)
        } else if sparql_upper.contains("ADD") {
            self.execute_add_operation(store, sparql)
        } else if sparql_upper.contains("CLEAR") {
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
            Ok(("UNKNOWN", 0, 0, vec!["default".to_string()]))
        }
    }
    /// Execute CLEAR operation
    fn execute_clear_operation(
        &self,
        store: &mut dyn CoreStore,
        sparql: &str,
    ) -> FusekiResult<(&'static str, usize, usize, Vec<String>)> {
        let sparql_upper = sparql.to_uppercase();
        if sparql_upper.contains("CLEAR ALL") {
            let all_quads = store.find_quads(None, None, None, None).map_err(|e| {
                FusekiError::update_execution(format!("Failed to query all quads: {e}"))
            })?;
            let mut deleted_count = 0;
            for quad in all_quads {
                if store.remove_quad(&quad).map_err(|e| {
                    FusekiError::update_execution(format!("Failed to remove quad: {e}"))
                })? {
                    deleted_count += 1;
                }
            }
            info!("Cleared all graphs: {} quads removed", deleted_count);
            return Ok(("CLEAR ALL", 0, deleted_count, vec!["*all*".to_string()]));
        }
        if sparql_upper.contains("CLEAR DEFAULT") || !sparql_upper.contains("CLEAR GRAPH <") {
            return self.clear_default_graph(store);
        }
        if let Some(graph_iri) = self.extract_graph_iri(sparql)? {
            return self.clear_named_graph(store, &graph_iri);
        }
        self.clear_default_graph(store)
    }
    /// Execute INSERT DATA operation
    fn execute_insert_data_operation(
        &self,
        store: &mut dyn CoreStore,
        sparql: &str,
    ) -> FusekiResult<(&'static str, usize, usize, Vec<String>)> {
        let data_block = self.extract_data_block(sparql, "INSERT DATA")?;
        let quads = self.parse_data_block(&data_block)?;
        let affected_graphs = self.extract_graph_names(&quads);
        let mut inserted_count = 0;
        for quad in quads {
            if store
                .insert_quad(quad)
                .map_err(|e| FusekiError::update_execution(format!("Failed to insert quad: {e}")))?
            {
                inserted_count += 1;
            }
        }
        Ok(("INSERT DATA", inserted_count, 0, affected_graphs))
    }
    /// Execute DELETE DATA operation
    fn execute_delete_data_operation(
        &self,
        store: &mut dyn CoreStore,
        sparql: &str,
    ) -> FusekiResult<(&'static str, usize, usize, Vec<String>)> {
        let data_block = self.extract_data_block(sparql, "DELETE DATA")?;
        let quads = self.parse_data_block(&data_block)?;
        let affected_graphs = self.extract_graph_names(&quads);
        let mut deleted_count = 0;
        for quad in quads {
            if store
                .remove_quad(&quad)
                .map_err(|e| FusekiError::update_execution(format!("Failed to remove quad: {e}")))?
            {
                deleted_count += 1;
            }
        }
        Ok(("DELETE DATA", 0, deleted_count, affected_graphs))
    }
    /// Execute simple INSERT operation
    fn execute_insert_operation(
        &self,
        store: &mut dyn CoreStore,
        sparql: &str,
    ) -> FusekiResult<(&'static str, usize, usize, Vec<String>)> {
        let data_block = self.extract_data_block(sparql, "INSERT")?;
        let quads = self.parse_data_block(&data_block)?;
        let affected_graphs = self.extract_graph_names(&quads);
        let mut inserted_count = 0;
        for quad in quads {
            if store
                .insert_quad(quad)
                .map_err(|e| FusekiError::update_execution(format!("Failed to insert quad: {e}")))?
            {
                inserted_count += 1;
            }
        }
        Ok(("INSERT", inserted_count, 0, affected_graphs))
    }
    /// Execute CREATE operation (SPARQL 1.1)
    /// Syntax: CREATE [SILENT] GRAPH <graphIRI>
    fn execute_create_operation(
        &self,
        _store: &mut dyn CoreStore,
        sparql: &str,
    ) -> FusekiResult<(&'static str, usize, usize, Vec<String>)> {
        info!("Executing CREATE operation: {}", sparql.trim());
        let sparql_upper = sparql.to_uppercase();
        let silent = sparql_upper.contains("SILENT");
        let graph_iri = self.extract_graph_iri_for_management(sparql, "CREATE")?;
        info!(
            "Graph '{}' marked for creation (graphs are created implicitly on data insertion)",
            graph_iri
        );
        if silent {
            Ok(("CREATE SILENT", 0, 0, vec![graph_iri]))
        } else {
            Ok(("CREATE", 0, 0, vec![graph_iri]))
        }
    }
    /// Execute DROP operation (SPARQL 1.1)
    /// Syntax: DROP [SILENT] (GRAPH <graphIRI> | DEFAULT | NAMED | ALL)
    fn execute_drop_operation(
        &self,
        store: &mut dyn CoreStore,
        sparql: &str,
    ) -> FusekiResult<(&'static str, usize, usize, Vec<String>)> {
        info!("Executing DROP operation: {}", sparql.trim());
        let sparql_upper = sparql.to_uppercase();
        let silent = sparql_upper.contains("SILENT");
        if sparql_upper.contains("DROP") && sparql_upper.contains("ALL") {
            let quad_count = store.len().map_err(|e| {
                FusekiError::update_execution(format!("Failed to get store size: {e}"))
            })?;
            store.clear_all().map_err(|e| {
                FusekiError::update_execution(format!("Failed to drop all graphs: {e}"))
            })?;
            info!("Dropped all graphs: {} quads removed", quad_count);
            return if silent {
                Ok(("DROP SILENT ALL", 0, quad_count, vec!["*all*".to_string()]))
            } else {
                Ok(("DROP ALL", 0, quad_count, vec!["*all*".to_string()]))
            };
        }
        if sparql_upper.contains("DROP") && sparql_upper.contains("DEFAULT") {
            return if silent {
                let result = self.clear_default_graph(store)?;
                Ok(("DROP SILENT DEFAULT", result.1, result.2, result.3))
            } else {
                let result = self.clear_default_graph(store)?;
                Ok(("DROP DEFAULT", result.1, result.2, result.3))
            };
        }
        if sparql_upper.contains("DROP") && sparql_upper.contains("NAMED") {
            let all_quads_raw = store.find_quads(None, None, None, None).map_err(|e| {
                FusekiError::update_execution(format!("Failed to query all quads: {e}"))
            })?;
            let all_quads: Vec<Quad> = all_quads_raw
                .into_iter()
                .filter(|quad| {
                    !matches!(
                        quad.graph_name(),
                        oxirs_core::model::GraphName::DefaultGraph
                    )
                })
                .collect();
            let deleted_count = all_quads.len();
            for quad in all_quads {
                store.remove_quad(&quad).map_err(|e| {
                    FusekiError::update_execution(format!("Failed to remove quad: {e}"))
                })?;
            }
            info!("Dropped all named graphs: {} quads removed", deleted_count);
            return if silent {
                Ok((
                    "DROP SILENT NAMED",
                    0,
                    deleted_count,
                    vec!["*named*".to_string()],
                ))
            } else {
                Ok(("DROP NAMED", 0, deleted_count, vec!["*named*".to_string()]))
            };
        }
        if let Ok(graph_iri) = self.extract_graph_iri_for_management(sparql, "DROP") {
            let result = self.clear_named_graph(store, &graph_iri)?;
            return if silent {
                Ok(("DROP SILENT GRAPH", result.1, result.2, result.3))
            } else {
                Ok(("DROP GRAPH", result.1, result.2, result.3))
            };
        }
        Err(FusekiError::update_execution(
            "Invalid DROP syntax: expected DROP [SILENT] (GRAPH <IRI> | DEFAULT | NAMED | ALL)"
                .to_string(),
        ))
    }
}
